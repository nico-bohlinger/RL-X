import os
import shutil
import json
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.mpo.flax.general_properties import GeneralProperties
from rl_x.algorithms.mpo.flax.policy import get_policy
from rl_x.algorithms.mpo.flax.critic import get_critic
from rl_x.algorithms.mpo.flax.dual_variables import DualVariables
from rl_x.algorithms.mpo.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.mpo.flax.rl_train_state import PolicyTrainState, CriticTrainState

rlx_logger = logging.getLogger("rl_x")


class MPO():
    def __init__(self, config, train_env, eval_env, run_path, writer):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.agent_learning_rate = config.algorithm.agent_learning_rate
        self.dual_learning_rate = config.algorithm.dual_learning_rate
        self.anneal_agent_learning_rate = config.algorithm.anneal_agent_learning_rate
        self.anneal_dual_learning_rate = config.algorithm.anneal_dual_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.actor_update_period = config.algorithm.actor_update_period
        self.target_network_update_period = config.algorithm.target_network_update_period
        self.gamma = config.algorithm.gamma
        self.n_steps = config.algorithm.n_steps
        self.optimize_every_n_steps = config.algorithm.optimize_every_n_steps
        self.action_sampling_number = config.algorithm.action_sampling_number
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.epsilon_non_parametric = config.algorithm.epsilon_non_parametric
        self.epsilon_parametric_mu = config.algorithm.epsilon_parametric_mu
        self.epsilon_parametric_sigma = config.algorithm.epsilon_parametric_sigma
        self.epsilon_penalty = config.algorithm.epsilon_penalty
        self.init_log_eta = config.algorithm.init_log_eta
        self.init_log_alpha_mean = config.algorithm.init_log_alpha_mean
        self.init_log_alpha_stddev = config.algorithm.init_log_alpha_stddev
        self.init_log_penalty_temperature = config.algorithm.init_log_penalty_temperature
        self.action_clipping = config.algorithm.action_clipping
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.nr_atoms = config.algorithm.nr_atoms
        self.float_epsilon = config.algorithm.float_epsilon
        self.min_log_temperature = config.algorithm.min_log_temperature
        self.min_log_alpha = config.algorithm.min_log_alpha
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, dual_key = jax.random.split(self.key, 4)

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        nr_actions = np.prod(self.train_env.single_action_space.shape).item()
        self.dual_variables = DualVariables(nr_actions, self.init_log_eta, self.init_log_alpha_mean, self.init_log_alpha_stddev, self.init_log_penalty_temperature)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.dual_variables.apply = jax.jit(self.dual_variables.apply)

        def agent_linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.agent_learning_rate * fraction
        
        def dual_linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.dual_learning_rate * fraction

        self.policy_learning_rate = agent_linear_schedule if self.anneal_agent_learning_rate else self.agent_learning_rate
        self.critic_learning_rate = agent_linear_schedule if self.anneal_agent_learning_rate else self.agent_learning_rate
        self.dual_learning_rate = dual_linear_schedule if self.anneal_dual_learning_rate else self.dual_learning_rate

        state = jnp.array([self.train_env.single_observation_space.sample()])
        action = jnp.array([self.train_env.single_action_space.sample()])

        self.policy_state = PolicyTrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            target_params=self.policy.init(policy_key, state),
            env_params=self.policy.init(policy_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate),
            )
        )

        self.critic_state = CriticTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, action),
            target_params=self.critic.init(critic_key, state, action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.critic_learning_rate),
            )
        )

        self.dual_variables_state = TrainState.create(
            apply_fn=self.dual_variables.apply,
            params=self.dual_variables.init(dual_key),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.dual_learning_rate),
            )
        )

        if self.enable_observation_normalization:
            self.observation_normalizer_state = {
                "running_mean": np.zeros((1, state.shape[-1]), dtype=np.float32),
                "running_var": np.ones((1, state.shape[-1]), dtype=np.float32),
                "running_std_dev": np.ones((1, state.shape[-1]), dtype=np.float32),
                "count": 0
            }
        else:
            self.observation_normalizer_state = None

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def _normalize_observations(self, observations, update=False, no_normalize_after_update=True):
        if not self.enable_observation_normalization:
            return observations

        if update:
            batch_mean = np.mean(observations, axis=0, keepdims=True)
            batch_var = np.var(observations, axis=0, keepdims=True)
            batch_count = observations.shape[0]

            new_count = self.observation_normalizer_state["count"] + batch_count
            delta = batch_mean - self.observation_normalizer_state["running_mean"]
            self.observation_normalizer_state["running_mean"] += delta * batch_count / new_count
            delta2 = batch_mean - self.observation_normalizer_state["running_mean"]
            m_a = self.observation_normalizer_state["running_var"] * self.observation_normalizer_state["count"]
            m_b = batch_var * batch_count
            m2 = m_a + m_b + np.square(delta2) * self.observation_normalizer_state["count"] * batch_count / new_count
            self.observation_normalizer_state["running_var"] = m2 / new_count
            self.observation_normalizer_state["running_std_dev"] = np.sqrt(self.observation_normalizer_state["running_var"])
            self.observation_normalizer_state["count"] = new_count

            if no_normalize_after_update:
                return

        return (observations - self.observation_normalizer_state["running_mean"]) / (self.observation_normalizer_state["running_std_dev"] + self.float_epsilon)
        
    
    def train(self):
        @jax.jit
        def get_action(policy_state: PolicyTrainState, state: np.ndarray, key: jax.random.PRNGKey):
            action_mean, action_std = self.policy.apply(policy_state.env_params, state)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            return action, key
        

        @jax.jit
        def update_actor(policy_state: PolicyTrainState):
            return policy_state.replace(env_params=policy_state.target_params)
        

        @jax.jit
        def get_deterministic_action(policy_state: PolicyTrainState, state: np.ndarray):
            action_mean, _ = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)
        


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(max(1, self.buffer_size // self.nr_envs)), self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.n_steps, self.gamma, self.rng)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None
        
        while global_step < self.total_timesteps:
            start_time = time.time()
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)


            # Acting
            dones_this_rollout = 0
            if global_step < self.learning_starts:
                processed_action = np.array([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                normalized_state = self._normalize_observations(state, update=False)
                action, self.key = get_action(self.policy_state, normalized_state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                    saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for key, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)

            replay_buffer.add(state, actual_next_state, action, reward, done, truncated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            iteration = global_step // self.nr_envs
            should_learning_start = global_step > self.learning_starts
            should_update_actor = should_learning_start and (iteration % self.actor_update_period == 0)
            should_optimize = should_learning_start and (iteration % self.optimize_every_n_steps == 0)
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0

            
            # Update actor
            if should_update_actor:
                self.policy_state = update_actor(self.policy_state)


            # TODO:
            # Optimizing
            if should_optimize:
                nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    eval_normalized_state = self._normalize_observations(eval_state, update=False)
                    eval_processed_action = get_deterministic_action(self.policy_state, eval_normalized_state)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(eval_processed_action))
                    eval_done = eval_terminated | eval_truncated
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i))
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)


            # Logging
            if should_log:
                self.start_logging(global_step)

                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_updates"] = nr_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes

                rollout_info_metrics = {}
                env_info_metrics = {}
                if step_info_collection:
                    info_names = list(step_info_collection.keys())
                    for info_name in info_names:
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(step_info_collection[info_name])
                        if mean_value == mean_value:  # Check if mean_value is NaN
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}

                self.end_logging()
            
            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time


    def get_buffer_mean(self, buffer):
        if len(buffer) > 0:
            return np.mean(buffer)
        else:
            return 0.0


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self):
        checkpoint = {
            "policy": self.policy_state,
            "critic": self.critic_state,
            "dual_variables": self.dual_variables_state,
            "observation_normalizer": self.observation_normalizer_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.best_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.best_model_file_name}.zip", f"{self.save_path}/{self.best_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = MPO(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "dual_variables": model.dual_variables_state,
            "observation_normalizer": model.observation_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.dual_variables_state = checkpoint["dual_variables"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: PolicyTrainState, state: np.ndarray):
            action_mean, _ = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                normalized_state = self._normalize_observations(state, update=False)
                processed_action = get_action(self.policy_state, normalized_state)
                state, reward, terminated, truncated, info = self.eval_env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
            return GeneralProperties
