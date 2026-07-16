import os
import shutil
import json
import logging
import time
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.simba.flax.general_properties import GeneralProperties
from rl_x.algorithms.simba.flax.policy import get_policy
from rl_x.algorithms.simba.flax.critic import get_critic
from rl_x.algorithms.simba.flax.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.simba.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.simba.flax import normalizer

rlx_logger = logging.getLogger("rl_x")


class Simba:
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
        self.learning_rate_init = config.algorithm.learning_rate_init
        self.learning_rate_end = config.algorithm.learning_rate_end
        self.weight_decay = config.algorithm.weight_decay
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.updates_per_step = config.algorithm.updates_per_step
        self.gamma = config.algorithm.gamma
        self.tau = config.algorithm.tau
        self.use_cdq = config.algorithm.use_cdq
        self.normalize_observation = config.algorithm.normalize_observation
        self.normalize_reward = config.algorithm.normalize_reward
        self.normalized_g_max = config.algorithm.normalized_g_max
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dim = self.as_shape[0]
        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high
        if config.algorithm.target_entropy == "auto":
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(config.algorithm.target_entropy)

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        rlx_logger.info(f"Using discount factor: {self.gamma}")

        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, entropy_coefficient_key = jax.random.split(self.key, 4)

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        self.entropy_coefficient = EntropyCoefficient(initial_value=config.algorithm.init_entropy_coefficient)

        total_updates = max(1, (self.total_timesteps // self.nr_envs) * self.updates_per_step)
        lr_schedule = optax.linear_schedule(
            init_value=self.learning_rate_init, end_value=self.learning_rate_end, transition_steps=total_updates,
        )

        dummy_obs = jnp.zeros((1,) + self.os_shape, dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.action_dim), dtype=jnp.float32)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_obs),
            tx=optax.inject_hyperparams(optax.adamw)(learning_rate=lr_schedule, weight_decay=self.weight_decay),
        )
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_action)
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.inject_hyperparams(optax.adamw)(learning_rate=lr_schedule, weight_decay=self.weight_decay),
        )
        self.target_critic_params = critic_params
        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule),
        )

        self.observation_normalizer_state = normalizer.init_observation_normalizer_state(self.os_shape)
        self.reward_normalizer_state = normalizer.init_reward_normalizer_state(self.nr_envs)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def train(self):
        @jax.jit
        def update_observation_normalizer(state, observation):
            return normalizer.update_observation_normalizer(state, observation)


        @jax.jit
        def update_reward_normalizer(state, reward, terminated, truncated):
            return normalizer.update_reward_normalizer(state, reward, terminated, truncated, self.gamma)


        @jax.jit
        def preprocess_observation(state, observation):
            if self.normalize_observation:
                return normalizer.normalize_observation(state, observation)
            return observation


        @jax.jit
        def get_action(policy_state, observation_normalizer_state, state, key):
            state = preprocess_observation(observation_normalizer_state, state)
            mean, log_std = self.policy.apply(policy_state.params, state)
            key, subkey = jax.random.split(key)
            action = self.policy.sample_action(mean, log_std, subkey)
            return action, key


        @jax.jit
        def get_deterministic_action(policy_state, observation_normalizer_state, state):
            state = preprocess_observation(observation_normalizer_state, state)
            mean, _ = self.policy.apply(policy_state.params, state)
            action = self.policy.deterministic_action(mean)
            return self.get_processed_action(action)


        @jax.jit
        def update_step(policy_state, critic_state, target_critic_params, entropy_coefficient_state, observation_normalizer_state, reward_normalizer_state, states, next_states, actions, rewards, terms, key):
            policy_key, critic_key = jax.random.split(key)
            states = preprocess_observation(observation_normalizer_state, states)
            next_states = preprocess_observation(observation_normalizer_state, next_states)
            if self.normalize_reward:
                rewards = normalizer.normalize_reward(reward_normalizer_state, rewards, self.normalized_g_max)

            def policy_loss_fn(policy_params):
                mean, log_std = self.policy.apply(policy_params, states)
                action, log_prob = self.policy.sample_and_log_prob(mean, log_std, policy_key)
                qs = self.critic.apply(critic_state.params, states, action)
                if self.use_cdq:
                    q = jnp.minimum(qs[0], qs[1])
                else:
                    q = qs
                entropy_coefficient = stop_gradient(self.entropy_coefficient.apply(entropy_coefficient_state.params))
                loss = jnp.mean(entropy_coefficient * log_prob - q)
                entropy = -jnp.mean(log_prob)
                return loss, (entropy, jnp.mean(q))

            (policy_loss_value, (entropy, policy_q_mean)), policy_gradients = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_state.params)
            policy_state = policy_state.apply_gradients(grads=policy_gradients)

            def entropy_coefficient_loss_fn(entropy_coefficient_params):
                entropy_coefficient = self.entropy_coefficient.apply(entropy_coefficient_params)
                loss = jnp.mean(entropy_coefficient * (stop_gradient(entropy) - self.target_entropy))
                return loss, entropy_coefficient

            (entropy_coefficient_loss_value, entropy_coefficient_value), entropy_coefficient_gradients = jax.value_and_grad(entropy_coefficient_loss_fn, has_aux=True)(entropy_coefficient_state.params)
            entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_coefficient_gradients)

            next_mean, next_log_std = self.policy.apply(policy_state.params, next_states)
            next_action, next_log_prob = self.policy.sample_and_log_prob(next_mean, next_log_std, critic_key)
            entropy_coefficient = self.entropy_coefficient.apply(entropy_coefficient_state.params)
            next_qs = self.critic.apply(target_critic_params, next_states, next_action)
            if self.use_cdq:
                next_q = jnp.minimum(next_qs[0], next_qs[1])
            else:
                next_q = next_qs
            target_q = stop_gradient(rewards + self.gamma * (1.0 - terms) * (next_q - entropy_coefficient * next_log_prob))

            def critic_loss_fn(critic_params):
                predicted_qs = self.critic.apply(critic_params, states, actions)
                if self.use_cdq:
                    critic_loss = ((predicted_qs[0] - target_q) ** 2 + (predicted_qs[1] - target_q) ** 2).mean()
                    q1_mean = predicted_qs[0].mean()
                    q2_mean = predicted_qs[1].mean()
                else:
                    critic_loss = ((predicted_qs - target_q) ** 2).mean()
                    q1_mean = q2_mean = predicted_qs.mean()
                return critic_loss, (q1_mean, q2_mean)

            (critic_loss_value, (q1_mean, q2_mean)), critic_gradients = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
            critic_state = critic_state.apply_gradients(grads=critic_gradients)
            new_target_params = jax.tree_util.tree_map(
                lambda p, tp: self.tau * p + (1.0 - self.tau) * tp,
                critic_state.params, target_critic_params,
            )

            metrics = {
                "loss/critic_loss": critic_loss_value,
                "loss/policy_loss": policy_loss_value,
                "loss/entropy_coefficient_loss": entropy_coefficient_loss_value,
                "entropy/entropy": entropy,
                "entropy/entropy_coefficient": entropy_coefficient_value,
                "q/target_q_mean": target_q.mean(),
                "q/q1_mean": q1_mean,
                "q/q2_mean": q2_mean,
                "q/policy_q_mean": policy_q_mean,
                "lr/learning_rate": critic_state.opt_state.hyperparams["learning_rate"],
            }
            return policy_state, critic_state, new_target_params, entropy_coefficient_state, metrics


        self.set_train_mode()

        replay_buffer = ReplayBuffer(self.buffer_size, self.nr_envs, self.os_shape, self.as_shape, self.rng)
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
            if self.normalize_observation:
                self.observation_normalizer_state = update_observation_normalizer(
                    self.observation_normalizer_state, jnp.asarray(state, dtype=jnp.float32),
                )
            if global_step < self.learning_starts:
                processed_action = np.stack([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                action, self.key = get_action(
                    self.policy_state, self.observation_normalizer_state, jnp.asarray(state, dtype=jnp.float32), self.key,
                )
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
            if self.normalize_reward:
                self.reward_normalizer_state = update_reward_normalizer(
                    self.reward_normalizer_state,
                    jnp.asarray(reward, dtype=jnp.float32),
                    jnp.asarray(terminated, dtype=jnp.bool_),
                    jnp.asarray(truncated, dtype=jnp.bool_),
                )
            replay_buffer.add(state, actual_next_state, action, reward, terminated)
            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout
            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_evaluate = self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing
            if should_optimize:
                for _ in range(self.updates_per_step):
                    states_b, next_states_b, actions_b, rewards_b, terms_b = replay_buffer.sample(self.batch_size)
                    self.key, upd_key = jax.random.split(self.key)
                    self.policy_state, self.critic_state, self.target_critic_params, self.entropy_coefficient_state, optimization_metrics = update_step(
                        self.policy_state, self.critic_state, self.target_critic_params, self.entropy_coefficient_state,
                        self.observation_normalizer_state, self.reward_normalizer_state,
                        jnp.asarray(states_b), jnp.asarray(next_states_b), jnp.asarray(actions_b),
                        jnp.asarray(rewards_b), jnp.asarray(terms_b), upd_key,
                    )
                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
                    nr_updates += 1
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    eval_action = get_deterministic_action(
                        self.policy_state, self.observation_normalizer_state, jnp.asarray(eval_state, dtype=jnp.float32),
                    )
                    eval_state, _, eval_term, eval_trunc, eval_info = self.eval_env.step(jax.device_get(eval_action))
                    eval_done = eval_term | eval_trunc
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
                mean_return = float(np.mean(saving_return_buffer))
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
                    for info_name, info_vals in step_info_collection.items():
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(info_vals)
                        if mean_value == mean_value:
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                time_metrics = {k: np.mean(v) for k, v in time_metrics_collection.items()}
                optimization_metrics = {k: float(np.mean(v)) for k, v in optimization_metrics_collection.items()}
                evaluation_metrics = {k: float(np.mean(v)) for k, v in evaluation_metrics_collection.items()}
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


    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)


    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self):
        checkpoint = {
            "policy": self.policy_state,
            "critic": self.critic_state,
            "target_critic_params": self.target_critic_params,
            "entropy_coefficient": self.entropy_coefficient_state,
            "observation_normalizer": self.observation_normalizer_state,
            "reward_normalizer": self.reward_normalizer_state,
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
        model = Simba(config, train_env, eval_env, run_path, writer)
        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "target_critic_params": model.target_critic_params,
            "entropy_coefficient": model.entropy_coefficient_state,
            "observation_normalizer": model.observation_normalizer_state,
            "reward_normalizer": model.reward_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.target_critic_params = checkpoint["target_critic_params"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]
        model.reward_normalizer_state = checkpoint["reward_normalizer"]
        shutil.rmtree(checkpoint_dir)
        return model


    def test(self, episodes):
        @jax.jit
        def act(state):
            if self.normalize_observation:
                state = normalizer.normalize_observation(self.observation_normalizer_state, state)
            mean, _ = self.policy.apply(self.policy_state.params, state)
            action = self.policy.deterministic_action(mean)
            return self.get_processed_action(action)

        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                action = act(jnp.asarray(state, dtype=jnp.float32))
                state, reward, terminated, truncated, info = self.eval_env.step(jax.device_get(action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
