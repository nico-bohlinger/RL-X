import os
import shutil
import json
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import flax
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.ddpg.flax.general_properties import GeneralProperties
from rl_x.algorithms.ddpg.flax.policy import get_policy
from rl_x.algorithms.ddpg.flax.critic import get_critic
from rl_x.algorithms.ddpg.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.ddpg.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class DDPG:
    def __init__(self, config, env, run_path, writer):
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.epsilon = config.algorithm.epsilon
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.env_as_low = env.single_action_space.low
        self.env_as_high = env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.env.single_observation_space.sample()])
        action = jnp.array([self.env.single_action_space.sample()])

        self.policy_state = RLTrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            target_params=self.policy.init(policy_key, state),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate)
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, action),
            target_params=self.critic.init(critic_key, state, action),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.q_learning_rate)
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
    
    def train(self):
        @jax.jit
        def get_action(policy_state: RLTrainState, state: np.ndarray, key: jax.random.PRNGKey):
            key, subkey = jax.random.split(key)
            mean_action = self.policy.apply(policy_state.params, state)
            action = mean_action + self.epsilon * jax.random.normal(subkey, mean_action.shape)
            action = jnp.clip(action, -1.0, 1.0)
            return action, key


        @jax.jit
        def update(
                policy_state: RLTrainState, critic_state: RLTrainState,
                states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray
            ):
            def loss_fn(policy_params: flax.core.FrozenDict, critic_params: flax.core.FrozenDict,
                        state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray
                ):
                # Critic loss
                next_action = self.policy.apply(policy_state.target_params, next_state)
                next_q_target = self.critic.apply(critic_state.target_params, next_state, next_action)
                y = reward + self.gamma * (1 - terminated) * next_q_target
                q = self.critic.apply(critic_params, state, action)
                q_loss = (q - y) ** 2

                # Policy loss
                current_action = self.policy.apply(policy_params, state)
                q = self.critic.apply(stop_gradient(critic_params), state, current_action)
                policy_loss = -q

                # Combine losses
                loss = q_loss + policy_loss

                # Create metrics
                metrics = {
                    "loss/q_loss": q_loss,
                    "loss/policy_loss": policy_loss,
                    "q_value/q_value": q,
                }

                return loss, (metrics)
            

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

            (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                policy_state.params, critic_state.params,
                states, next_states, actions, rewards, terminations)

            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            critic_state = critic_state.apply_gradients(grads=critic_gradients)

            # Update targets
            critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))
            policy_state = policy_state.replace(target_params=optax.incremental_update(policy_state.params, policy_state.target_params, self.tau))

            metrics["lr/learning_rate"] = policy_state.opt_state.hyperparams["learning_rate"]
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

            return policy_state, critic_state, metrics
        

        @jax.jit
        def get_deterministic_action(policy_state: RLTrainState, state: np.ndarray):
            mean_action = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(mean_action)


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.single_observation_space.shape, self.env.single_action_space.shape, self.rng)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            dones_this_rollout = 0
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                action, self.key = get_action(self.policy_state, state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = self.env.get_final_observation_at_index(info, i)
                    saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for key, info_value in self.env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing - Prepare batches
            if should_optimize:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-function and policy
            if should_optimize:
                self.policy_state, self.critic_state, optimization_metrics = update(self.policy_state, self.critic_state, batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations)
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_updates += 1
            
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                while True:
                    processed_action = get_deterministic_action(self.policy_state, state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                state, _ = self.env.reset()
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
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)

            time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - start_time))


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
            "critic": self.critic_state          
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


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = DDPG(config, env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: RLTrainState, state: np.ndarray):
            mean_action = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(mean_action)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while not done:
                processed_action = get_action(self.policy_state, state)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
