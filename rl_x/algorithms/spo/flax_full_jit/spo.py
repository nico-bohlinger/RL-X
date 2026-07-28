import os
import shutil
import json
from copy import deepcopy
import logging
import time
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.spo.flax_full_jit.critic import Critic
from rl_x.algorithms.spo.flax_full_jit.policy import get_policy
from rl_x.algorithms.spo.flax_full_jit import observation_normalizer
from rl_x.algorithms.spo.flax_full_jit import reward_normalizer

rlx_logger = logging.getLogger("rl_x")


class SPO:
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
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.render_callback_type = getattr(config.environment, "render_callback_type", "io_callback")
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.spo_epsilon = config.algorithm.spo_epsilon
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.clip_value_loss = config.algorithm.clip_value_loss
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.normalize_observation = config.algorithm.normalize_observation
        self.normalize_reward = config.algorithm.normalize_reward
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active

        self.batch_size = self.nr_envs * self.nr_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if self.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * self.nr_updates
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon
        critic_observation_indices = getattr(self.train_env, "critic_observation_indices", jnp.arange(self.os_shape[0]))

        if self.nr_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("The rollout batch size must be divisible by the minibatch size.")
        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of the rollout batch size.")
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet.")
        if self.spo_epsilon <= 0.0:
            raise ValueError("SPO epsilon must be positive.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        env_state = self.train_env.reset(jax.random.split(reset_key, 1), False)
        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = Critic(critic_observation_indices)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        optimizer = optax.chain(optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate))
        self.policy_state = TrainState.create(apply_fn=self.policy.apply, params=self.policy.init(policy_key, env_state.next_observation), tx=optimizer)
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.nr_envs, self.os_shape)
        self.reward_normalizer_state = reward_normalizer.init_reward_normalizer_state(self.nr_envs)
        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=self.critic.init(critic_key, env_state.next_observation), tx=optimizer)

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key)
            env_state = self.train_env.reset(jax.random.split(reset_key, self.nr_envs), False)
            policy_state = self.policy_state
            critic_state = self.critic_state
            normalizer_state = self.observation_normalizer_state
            reward_normalizer_state = self.reward_normalizer_state

            def multi_iteration(carry, multi_iteration_step):
                (policy_state, critic_state, normalizer_state, reward_normalizer_state, env_state, key) = carry

                def learning_iteration(carry, learning_iteration_step):
                    (policy_state, critic_state, normalizer_state, reward_normalizer_state, env_state, key) = carry

                    # Acting
                    def rollout_step(carry, _):
                        env_state, normalizer_state, key = carry
                        observation = env_state.next_observation
                        if self.normalize_observation:
                            normalizer_state = observation_normalizer.update_observation_normalizer(normalizer_state, observation)
                        normalized_observation = observation_normalizer.normalize_observation(normalizer_state, observation) if self.normalize_observation else observation
                        key, action_key = jax.random.split(key)
                        action_mean, action_logstd = self.policy.apply(policy_state.params, normalized_observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(action_key, action_mean.shape)
                        log_probability = jnp.sum(-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd, axis=-1)
                        value = self.critic.apply(critic_state.params, normalized_observation).squeeze(-1)
                        env_state = self.train_env.step(env_state, self.get_processed_action(action))
                        normalized_next_observation = observation_normalizer.normalize_observation(normalizer_state, env_state.actual_next_observation) if self.normalize_observation else env_state.actual_next_observation
                        transition = normalized_observation, normalized_next_observation, action, log_probability, env_state.reward, value, env_state.terminated, env_state.truncated, env_state.info
                        if self.render:
                            if self.render_callback_type == "debug_callback":
                                jax.debug.callback(self.train_env.render, env_state)
                            else:
                                env_state = jax.experimental.io_callback(self.train_env.render, env_state, env_state)
                        return (env_state, normalizer_state, key), transition

                    (env_state, normalizer_state, key), batch = jax.lax.scan(rollout_step, (env_state, normalizer_state, key), None, self.nr_steps)
                    (states, next_states, actions, behavior_log_probabilities, rewards, values, terminations, truncations, infos) = batch
                    next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)

                    # Calculating advantages and returns
                    if self.normalize_reward:
                        (reward_normalizer_state, rewards) = reward_normalizer.normalize_reward(reward_normalizer_state, rewards, terminations, truncations, self.gamma)

                    def advantage_step(next_advantage, inputs):
                        reward, value, next_value, terminated, truncated = inputs
                        delta = reward + self.gamma * (1.0 - terminated) * next_value - value
                        continuation = (1.0 - terminated) * (1.0 - truncated)
                        advantage = delta + self.gamma * self.gae_lambda * continuation * next_advantage
                        return advantage, advantage

                    _, advantages = jax.lax.scan(advantage_step, jnp.zeros_like(values[-1]), (rewards, values, next_values, terminations, truncations), reverse=True)
                    returns = advantages + values
                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_behavior_log_probabilities = behavior_log_probabilities.reshape(-1)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_values = values.reshape(-1)

                    # Optimizing
                    def loss_fn(policy_params, critic_params, state_b, action_b, behavior_log_probability_b, advantage_b, return_b, behavior_value_b):
                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                        action_std = jnp.exp(action_logstd)
                        current_log_probability = jnp.sum(-0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd, axis=-1)
                        log_ratio = current_log_probability - behavior_log_probability_b
                        ratio = jnp.exp(log_ratio)
                        normalized_advantage = (advantage_b - jnp.mean(advantage_b)) / (jnp.std(advantage_b) + 1e-8)
                        ratio_deviation_penalty = jnp.abs(normalized_advantage) * (ratio - 1.0) ** 2 / (2.0 * self.spo_epsilon)
                        policy_loss = jnp.mean(-normalized_advantage * ratio + ratio_deviation_penalty)
                        entropy = jnp.sum(action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)
                        value = self.critic.apply(critic_params, state_b).squeeze(-1)
                        if self.clip_value_loss:
                            unclipped_value_loss = (value - return_b) ** 2
                            clipped_value = behavior_value_b + jnp.clip(value - behavior_value_b, -self.spo_epsilon, self.spo_epsilon)
                            clipped_value_loss = (clipped_value - return_b) ** 2
                            critic_loss = 0.5 * jnp.mean(jnp.maximum(unclipped_value_loss, clipped_value_loss))
                        else:
                            critic_loss = 0.5 * jnp.mean((value - return_b) ** 2)
                        total_loss = policy_loss - self.entropy_coef * jnp.mean(entropy) + self.critic_coef * critic_loss
                        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                        metrics = {
                            "loss/policy_gradient_loss": policy_loss,
                            "loss/ratio_deviation_penalty": jnp.mean(ratio_deviation_penalty),
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": jnp.mean(entropy),
                            "policy_ratio/approx_kl": approx_kl,
                            "policy_ratio/mean": jnp.mean(ratio),
                            "policy_ratio/min": jnp.min(ratio),
                            "policy_ratio/max": jnp.max(ratio),
                            "advantage/normalized_mean": jnp.mean(normalized_advantage),
                            "advantage/normalized_std": jnp.std(normalized_advantage),
                        }
                        return total_loss, metrics

                    grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
                    key, shuffle_key = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(shuffle_key, batch_indices, axis=1, independent=True).reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry
                        ((_, metrics), (policy_gradients, critic_gradients)) = grad_loss_fn(policy_state.params, critic_state.params, batch_states[minibatch_indices], batch_actions[minibatch_indices], batch_behavior_log_probabilities[minibatch_indices], batch_advantages[minibatch_indices], batch_returns[minibatch_indices], batch_values[minibatch_indices])
                        combined_gradient_norm = jnp.sqrt(optax.global_norm(policy_gradients) ** 2 + optax.global_norm(critic_gradients) ** 2)
                        gradient_scale = jnp.minimum(1.0, self.max_grad_norm / (combined_gradient_norm + 1e-6))
                        policy_state = policy_state.apply_gradients(grads=tree.map_structure(lambda gradient: gradient * gradient_scale, policy_gradients))
                        critic_state = critic_state.apply_gradients(grads=tree.map_structure(lambda gradient: gradient * gradient_scale, critic_gradients))
                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                        return (policy_state, critic_state), metrics

                    ((policy_state, critic_state), optimization_metrics) = jax.lax.scan(minibatch_update, (policy_state, critic_state), batch_indices)
                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[0].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1.0 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    combined_metrics = tree.map_structure(jnp.mean, {**infos, **optimization_metrics})

                    # Logging
                    def callback(callback_carry):
                        (metrics, learning_iteration_step, multi_iteration_step, parallel_seed_id) = callback_carry
                        current_time = time.time()
                        metrics["time/sps"] = int(self.batch_size / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        combined_step = multi_iteration_step * self.nr_updates_per_multi_learning_iteration + learning_iteration_step + 1
                        global_step = int(combined_step.item() * self.batch_size)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = combined_step.item() * self.nr_epochs * self.nr_minibatches
                        self.start_logging(global_step)
                        for name, value in metrics.items():
                            self.log(name, np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, multi_iteration_step, parallel_seed_id))
                    return (policy_state, critic_state, normalizer_state, reward_normalizer_state, env_state, key), None

                carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, normalizer_state, reward_normalizer_state, env_state, key), jnp.arange(self.nr_updates_per_multi_learning_iteration))
                (policy_state, critic_state, normalizer_state, reward_normalizer_state, env_state, key) = carry

                # Evaluating
                if self.evaluation_active:
                    def eval_rollout(eval_carry, _):
                        eval_env_state, key = eval_carry
                        observation = eval_env_state.next_observation
                        if self.normalize_observation:
                            observation = observation_normalizer.normalize_observation(normalizer_state, observation)
                        action_mean, unused_action_logstd = self.policy.apply(policy_state.params, observation)
                        eval_env_state = self.eval_env.step(eval_env_state, self.get_processed_action(action_mean))
                        return (eval_env_state, key), None

                    key, reset_key = jax.random.split(key)
                    eval_env_state = self.eval_env.reset(jax.random.split(reset_key, self.nr_envs), True)
                    (eval_env_state, key), _ = jax.lax.scan(eval_rollout, (eval_env_state, key), None, self.horizon)
                    evaluation_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    def callback(metrics_and_step):
                        metrics, combined_step = metrics_and_step
                        global_step = int(combined_step.item() * self.batch_size)
                        self.start_logging(global_step)
                        for name, value in metrics.items():
                            self.log(name, np.asarray(value), global_step)
                        self.end_logging()

                    combined_step = (multi_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration
                    jax.debug.callback(callback, (evaluation_metrics, combined_step))

                # Saving
                if self.save_model:
                    jax.debug.callback(self.save, policy_state, critic_state, normalizer_state, reward_normalizer_state)
                return (policy_state, critic_state, normalizer_state, reward_normalizer_state, env_state, key), None

            jax.lax.scan(multi_iteration, (policy_state, critic_state, normalizer_state, reward_normalizer_state, env_state, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max(time.time() - start_time for start_time in self.start_time):.2f} s")


    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            value = np.format_float_positional(value, trim="-")
            rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, policy_state, critic_state, normalizer_state, reward_normalizer_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "observation_normalizer": normalizer_state,
            "reward_normalizer": reward_normalizer_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as stream:
            json.dump(self.config.algorithm.to_dict(), stream)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")
        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    @staticmethod
    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        split_path = config.runner.load_model.split("/")
        checkpoint_directory = os.path.abspath("/".join(split_path[:-1]))
        checkpoint_file_name = split_path[-1]
        shutil.unpack_archive(f"{checkpoint_directory}/{checkpoint_file_name}", f"{checkpoint_directory}/tmp", "zip")
        checkpoint_directory = f"{checkpoint_directory}/tmp"
        with open(f"{checkpoint_directory}/config_algorithm.json") as stream:
            loaded_algorithm_config = json.load(stream)
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = SPO(config, train_env, eval_env, run_path, writer)
        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "observation_normalizer": model.observation_normalizer_state,
            "reward_normalizer": model.reward_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_directory, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]
        model.reward_normalizer_state = checkpoint["reward_normalizer"]
        shutil.rmtree(checkpoint_directory)
        return model


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")
        key, reset_key = jax.random.split(self.key)
        env_state = self.eval_env.reset(jax.random.split(reset_key, self.nr_envs), True)

        @jax.jit
        def rollout(env_state, key):
            def step(carry, _):
                env_state, key = carry
                action_mean, _ = self.policy.apply(self.policy_state.params, (observation_normalizer.normalize_observation(self.observation_normalizer_state, env_state.next_observation) if self.normalize_observation else env_state.next_observation))
                env_state = self.eval_env.step(env_state, self.get_processed_action(action_mean))
                return (env_state, key), None

            return jax.lax.scan(step, (env_state, key), None, self.horizon)[0]

        while True:
            env_state, key = rollout(env_state, key)
            self.eval_env.render(env_state)
