import os
import shutil
import json
from functools import partial
import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.dime.flax.general_properties import GeneralProperties
from rl_x.algorithms.dime.flax.policy import get_policy
from rl_x.algorithms.dime.flax.critic import get_critic
from rl_x.algorithms.dime.flax.entropy_coefficient import get_entropy_coefficient
from rl_x.algorithms.dime.flax.rl_train_state import RLTrainState
from rl_x.algorithms.dime.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.dime.flax import observation_normalizer

rlx_logger = logging.getLogger("rl_x")


class DIME:
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
        self.actor_learning_rate = config.algorithm.actor_learning_rate
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.entropy_learning_rate = config.algorithm.entropy_learning_rate
        self.adam_beta1 = config.algorithm.adam_beta1
        self.adam_beta2 = config.algorithm.adam_beta2
        self.batch_size = config.algorithm.batch_size
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.updates_per_step = config.algorithm.updates_per_step
        self.policy_delay = config.algorithm.policy_delay
        self.gamma = config.algorithm.gamma
        self.policy_tau = config.algorithm.policy_tau
        self.nr_critics = config.algorithm.nr_critics
        self.nr_atoms = config.algorithm.nr_atoms
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.critic_entropy_coefficient = config.algorithm.critic_entropy_coefficient
        self.diffusion_steps = config.algorithm.diffusion_steps
        self.prior_std = config.algorithm.prior_std
        self.minimum_timestep = config.algorithm.minimum_timestep
        self.cosine_schedule_offset = config.algorithm.cosine_schedule_offset
        self.target_entropy_per_action_dimension = config.algorithm.target_entropy_per_action_dimension
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.normalizer_epsilon = config.algorithm.normalizer_epsilon
        self.action_rescaling = config.algorithm.action_rescaling
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dimension = self.as_shape[0]
        self.action_low = jnp.asarray(self.train_env.single_action_space.low)
        self.action_high = jnp.asarray(self.train_env.single_action_space.high)
        self.support = jnp.linspace(self.v_min, self.v_max, self.nr_atoms)
        self.nr_iterations = self.total_timesteps // self.nr_envs
        self.target_entropy = self.target_entropy_per_action_dimension * self.action_dimension

        if self.total_timesteps < self.nr_envs:
            raise ValueError("Total timesteps must contain one environment step.")
        if self.buffer_size < self.learning_starts:
            raise ValueError("Replay capacity must reach the learning-start gate.")
        if self.batch_size < 1 or self.updates_per_step < 1:
            raise ValueError("Batch and update counts must be positive.")
        if self.policy_delay < 1 or self.diffusion_steps < 1:
            raise ValueError("Policy delay and diffusion steps must be positive.")
        if self.nr_critics != 2:
            raise ValueError("The reference DIME categorical update requires two critics.")
        if self.nr_atoms < 2 or self.v_max <= self.v_min:
            raise ValueError("Categorical critic support is invalid.")
        if self.prior_std <= 0.0 or self.minimum_timestep <= 0.0:
            raise ValueError("Prior standard deviation and timestep must be positive.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        (self.key, actor_key, critic_key, entropy_key) = jax.random.split(self.key, 4)
        dummy_observation = jnp.asarray([self.train_env.single_observation_space.sample()])
        dummy_action = jnp.zeros(dummy_observation.shape[:-1] + self.as_shape)
        dummy_timestep = jnp.zeros(dummy_observation.shape[:-1] + (1,))

        self.actor = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        self.entropy_coefficient = get_entropy_coefficient(config)

        actor_params = self.actor.init(actor_key, dummy_observation, dummy_action, dummy_timestep)
        critic_variables = self.critic.init({"params": critic_key, "batch_stats": critic_key,}, dummy_observation, dummy_action, False)
        self.actor_state = TrainState.create(apply_fn=self.actor.apply, params=actor_params, tx=optax.chain(optax.zero_nans(), optax.clip_by_global_norm(self.max_grad_norm), optax.adam(self.actor_learning_rate, b1=self.adam_beta1, b2=self.adam_beta2)))
        self.target_actor_state = TrainState.create(apply_fn=self.actor.apply, params=actor_params, tx=optax.set_to_zero())
        self.critic_state = RLTrainState.create(apply_fn=self.critic.apply, params=critic_variables["params"], batch_stats=critic_variables["batch_stats"], tx=optax.adam(self.critic_learning_rate, b1=self.adam_beta1, b2=self.adam_beta2))
        self.entropy_state = TrainState.create(apply_fn=self.entropy_coefficient.apply, params=self.entropy_coefficient.init(entropy_key), tx=optax.adam(self.entropy_learning_rate))
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def normalize(self, normalizer_state, observation):
        if self.enable_observation_normalization:
            return observation_normalizer.normalize_observation(normalizer_state, observation, self.normalizer_epsilon)
        return observation


    def sample_action(self, actor_params, normalized_observation, key, deterministic=False):
        key, prior_key, noise_key = jax.random.split(key, 3)
        initial_action = self.prior_std * jax.random.normal(prior_key, normalized_observation.shape[:-1] + self.as_shape)
        noise_path = jax.random.normal(noise_key, (self.diffusion_steps,) + initial_action.shape)
        if deterministic:
            noise_path = jnp.zeros_like(noise_path)
        base_timestep = jax.nn.softplus(actor_params["params"]["log_timestep"])
        friction = jax.nn.softplus(actor_params["params"]["log_friction"])

        def diffusion_step(carry, inputs):
            action, log_ratio = carry
            step, noise = inputs
            timestep = jnp.full(normalized_observation.shape[:-1] + (1,), step)
            reverse_time = (self.diffusion_steps - step) / self.diffusion_steps
            offset = 1.0 + self.cosine_schedule_offset
            timestep_delta = base_timestep * ((1.0 - self.minimum_timestep) * jnp.cos(0.5 * jnp.pi * (offset - reverse_time) / offset) ** 2 + self.minimum_timestep)
            variance_time = timestep_delta / friction
            transition_std = jnp.sqrt(2.0 * variance_time)
            prior_score = -action / self.prior_std ** 2
            control = self.actor.apply(actor_params, normalized_observation, action, timestep)
            forward_mean = action + variance_time * (prior_score + control)
            next_action = forward_mean + transition_std * noise
            backward_mean = next_action + variance_time * (-next_action / self.prior_std ** 2)
            forward_log_probability = jnp.sum(-0.5 * ((next_action - forward_mean) / transition_std) ** 2 - jnp.log(transition_std) - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)
            backward_log_probability = jnp.sum(-0.5 * ((action - backward_mean) / transition_std) ** 2 - jnp.log(transition_std) - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)
            return (next_action, log_ratio + backward_log_probability - forward_log_probability), next_action

        (final_latent, log_ratio), latent_path = jax.lax.scan(diffusion_step, (initial_action, jnp.zeros(initial_action.shape[:-1])), (jnp.arange(self.diffusion_steps, dtype=jnp.float32), noise_path))
        normalized_action = jnp.tanh(final_latent)
        tanh_log_determinant = jnp.sum(jnp.log(1.0 - normalized_action ** 2 + 1e-6), axis=-1)
        running_cost = -(log_ratio + tanh_log_determinant)
        terminal_cost = jnp.sum(-0.5 * (initial_action / self.prior_std) ** 2 - jnp.log(self.prior_std) - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)
        latent_path = jnp.moveaxis(latent_path, 0, -2)
        return key, normalized_action, running_cost, jnp.zeros_like(running_cost), terminal_cost, latent_path


    def project_distribution(self, next_distribution, reward, terminated, entropy_bonus):
        target_support = jnp.clip(reward[..., None] + self.gamma * (1.0 - terminated[..., None]) * (self.support - entropy_bonus[..., None]), self.v_min, self.v_max)
        atom_delta = (self.v_max - self.v_min) / (self.nr_atoms - 1)
        position = (target_support - self.v_min) / atom_delta
        lower = jnp.floor(position).astype(jnp.int32)
        upper = jnp.ceil(position).astype(jnp.int32)
        lower = jnp.where((upper > 0) & (lower == upper), lower - 1, lower)
        upper = jnp.where((lower < self.nr_atoms - 1) & (lower == upper), upper + 1, upper)
        batch_offset = jnp.arange(reward.shape[0])[:, None] * self.nr_atoms
        projected = jnp.zeros_like(next_distribution).reshape(-1)
        projected = projected.at[(lower + batch_offset).reshape(-1)].add((next_distribution * (upper.astype(jnp.float32) - position)).reshape(-1))
        projected = projected.at[(upper + batch_offset).reshape(-1)].add((next_distribution * (position - lower.astype(jnp.float32))).reshape(-1))
        return projected.reshape(next_distribution.shape)


    def train(self):
        @partial(jax.jit, static_argnames=("deterministic",))
        def get_action(actor_params, normalizer_state, observation, key, deterministic=False):
            normalized_observation = self.normalize(normalizer_state, observation)
            return self.sample_action(actor_params, normalized_observation, key, deterministic)


        @jax.jit
        def update(actor_state, target_actor_state, critic_state, entropy_state, states, next_states, actions, rewards, terminations, update_count, key):
            key, next_action_key, actor_key = jax.random.split(key, 3)
            (unused_key, next_action, next_running_cost, next_stochastic_cost, next_terminal_cost, unused_path) = self.sample_action(target_actor_state.params, next_states, next_action_key)
            entropy_coefficient = entropy_state.apply_fn(entropy_state.params)
            entropy_bonus = entropy_coefficient * (next_running_cost + next_stochastic_cost + next_terminal_cost)

            def critic_loss_fn(critic_params, critic_batch_stats):
                current_and_next_distribution, critic_state_update = self.critic.apply({"params": critic_params, "batch_stats": critic_batch_stats}, jnp.concatenate([states, next_states], axis=0), jnp.concatenate([actions, jax.lax.stop_gradient(next_action)], axis=0), True, mutable=["batch_stats"])
                current_distribution, next_distribution = jnp.split(current_and_next_distribution, 2, axis=1)
                target_distribution = jax.lax.stop_gradient((self.project_distribution(next_distribution[0], rewards, terminations, entropy_bonus) + self.project_distribution(next_distribution[1], rewards, terminations, entropy_bonus)) / 2.0)
                cross_entropy = -jnp.sum(jnp.mean(jnp.sum(target_distribution[None] * jnp.log(current_distribution + 1e-15), axis=-1), axis=-1))
                distribution_entropy = jnp.sum(jnp.mean(jnp.sum(current_distribution * jnp.log(current_distribution + 1e-15), axis=-1), axis=-1))
                loss = cross_entropy + self.critic_entropy_coefficient * distribution_entropy
                metrics = {
                    "loss/critic_loss": loss,
                    "q/target_mean": jnp.mean(jnp.sum(target_distribution * self.support, axis=-1)),
                    "q/current_mean": jnp.mean(jnp.sum(current_distribution * self.support, axis=-1)),
                    "q/distribution_entropy": -jnp.mean(jnp.sum(current_distribution * jnp.log(current_distribution + 1e-15), axis=-1)),
                    "critic_state_update": critic_state_update,
                }
                return loss, metrics

            (unused_critic_loss, critic_metrics), critic_gradients = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=True)(critic_state.params, critic_state.batch_stats)
            critic_state_update = critic_metrics.pop("critic_state_update")
            critic_state = critic_state.apply_gradients(grads=critic_gradients)
            critic_state = critic_state.replace(batch_stats=critic_state_update["batch_stats"])

            def actor_and_temperature_update(update_carry):
                actor_state, target_actor_state, entropy_state = update_carry

                def actor_loss_fn(actor_params):
                    (unused_key, sampled_action, running_cost, stochastic_cost, terminal_cost, latent_path) = self.sample_action(actor_params, states, actor_key)
                    q_distribution = self.critic.apply({"params": critic_state.params, "batch_stats": critic_state.batch_stats}, states, sampled_action, False)
                    q_value = jnp.mean(jnp.sum(q_distribution * self.support, axis=-1), axis=0)
                    path_cost = running_cost + stochastic_cost + terminal_cost
                    loss = jnp.mean(-q_value + jax.lax.stop_gradient(entropy_coefficient) * path_cost)
                    return loss, {"loss/actor_loss": loss, "entropy/running_cost": jnp.mean(running_cost), "entropy/stochastic_cost": jnp.mean(stochastic_cost), "entropy/terminal_cost": jnp.mean(terminal_cost), "policy/latent_abs_max": jnp.max(jnp.abs(latent_path)), "q/policy_mean": jnp.mean(q_value)}

                (unused_actor_loss, actor_metrics), actor_gradients = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
                actor_state = actor_state.apply_gradients(grads=actor_gradients)
                target_actor_state = target_actor_state.replace(params=optax.incremental_update(actor_state.params, target_actor_state.params, self.policy_tau))

                def entropy_loss_fn(entropy_params):
                    coefficient = self.entropy_coefficient.apply(entropy_params)
                    return -coefficient * jax.lax.stop_gradient(actor_metrics["entropy/running_cost"] - self.target_entropy)

                entropy_loss, entropy_gradients = jax.value_and_grad(entropy_loss_fn)(entropy_state.params)
                entropy_state = entropy_state.apply_gradients(grads=entropy_gradients)
                actor_metrics["loss/entropy_coefficient_loss"] = entropy_loss
                actor_metrics["entropy/coefficient"] = entropy_state.apply_fn(entropy_state.params)
                actor_metrics["gradients/actor_grad_norm"] = optax.global_norm(actor_gradients)
                actor_metrics["actor/update_active"] = jnp.ones(())
                actor_metrics["entropy/target_mismatch"] = actor_metrics["entropy/running_cost"] - self.target_entropy
                return (actor_state, target_actor_state, entropy_state), actor_metrics

            def skip_actor_update(update_carry):
                actor_state, target_actor_state, entropy_state = update_carry
                return update_carry, {"loss/actor_loss": jnp.zeros(()), "entropy/running_cost": jnp.zeros(()), "entropy/stochastic_cost": jnp.zeros(()), "entropy/terminal_cost": jnp.zeros(()), "policy/latent_abs_max": jnp.zeros(()), "q/policy_mean": jnp.zeros(()), "loss/entropy_coefficient_loss": jnp.zeros(()), "entropy/coefficient": entropy_state.apply_fn(entropy_state.params), "gradients/actor_grad_norm": jnp.zeros(()), "actor/update_active": jnp.zeros(()), "entropy/target_mismatch": jnp.zeros(())}

            (actor_state, target_actor_state, entropy_state), actor_metrics = jax.lax.cond((update_count + 1) % self.policy_delay == 0, actor_and_temperature_update, skip_actor_update, (actor_state, target_actor_state, entropy_state))
            metrics = {
                **critic_metrics,
                **actor_metrics,
                "gradients/critic_grad_norm": optax.global_norm(critic_gradients),
            }
            return actor_state, target_actor_state, critic_state, entropy_state, metrics, update_count + 1, key


        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.os_shape, self.as_shape, np.random.default_rng(self.seed))
        state, unused_info = self.train_env.reset()
        global_step = 0
        update_count = 0
        update_budget = 0.0
        metrics_collection = {}
        step_info_collection = {}
        logging_start_time = time.time()
        while global_step < self.total_timesteps:
            # Acting
            if self.enable_observation_normalization:
                self.observation_normalizer_state = observation_normalizer.update_observation_normalizer(self.observation_normalizer_state, state)
            normalized_state = self.normalize(self.observation_normalizer_state, state)
            self.key, action, unused_running_cost, unused_stochastic_cost, unused_terminal_cost, unused_path = get_action(self.actor_state.params, self.observation_normalizer_state, state, self.key)
            if self.action_rescaling:
                processed_action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
            else:
                processed_action = action
            next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
            actual_next_state = next_state.copy()
            for index, done in enumerate(terminated | truncated):
                if done:
                    actual_next_state[index] = np.asarray(self.train_env.get_final_observation_at_index(info, index))
            normalized_next_state = self.normalize(self.observation_normalizer_state, actual_next_state)
            replay_buffer.add(normalized_state, normalized_next_state, action, reward, terminated)
            for name, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(name, []).extend(info_value)
            state = next_state
            global_step += self.nr_envs

            # Updating
            if global_step >= self.learning_starts:
                update_budget += self.updates_per_step * self.nr_envs / self.batch_size
                nr_updates = int(update_budget)
                update_budget -= nr_updates
                for unused_update in range(nr_updates):
                    batch = replay_buffer.sample(self.batch_size)
                    (self.actor_state, self.target_actor_state, self.critic_state, self.entropy_state, metrics, update_count, self.key) = update(self.actor_state, self.target_actor_state, self.critic_state, self.entropy_state, *batch, update_count, self.key)
                    for name, value in metrics.items():
                        metrics_collection.setdefault(name, []).append(value)

            # Evaluating
            if self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0:
                eval_state, unused_info = self.eval_env.reset()
                completed_episodes = 0
                while completed_episodes < self.evaluation_episodes:
                    self.key, eval_action, unused_running_cost, unused_stochastic_cost, unused_terminal_cost, unused_path = get_action(self.actor_state.params, self.observation_normalizer_state, eval_state, self.key, True)
                    if self.action_rescaling:
                        eval_action = self.action_low + 0.5 * (eval_action + 1.0) * (self.action_high - self.action_low)
                    eval_state, unused_reward, eval_terminated, eval_truncated, unused_info = self.eval_env.step(jax.device_get(eval_action))
                    completed_episodes += int(np.sum(eval_terminated | eval_truncated))

            # Saving
            if self.save_model and global_step >= self.total_timesteps:
                self.save(self.actor_state, self.critic_state, self.entropy_state, self.observation_normalizer_state)

            # Logging
            if global_step % self.logging_frequency == 0:
                metrics = {name: np.mean(jax.device_get(values)) for name, values in metrics_collection.items()}
                for name, values in step_info_collection.items():
                    metric_group = "rollout" if name in ["episode_return", "episode_length"] else "env_info"
                    metrics[f"{metric_group}/{name}"] = np.mean(values)
                metrics["replay/fill_fraction"] = replay_buffer.size / replay_buffer.capacity
                metrics["time/sps"] = self.logging_frequency / (time.time() - logging_start_time)
                metrics["steps/nr_env_steps"] = global_step
                metrics["steps/nr_updates"] = update_count
                self.start_logging(global_step)
                for name, value in metrics.items():
                    self.log(name, value, global_step)
                self.end_logging()
                metrics_collection = {}
                step_info_collection = {}
                logging_start_time = time.time()


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
            self.wandb_log_cache = {
                "global_step": int(step)
            }
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, actor_state, critic_state, entropy_state, normalizer_state):
        checkpoint = {
            "actor": actor_state,
            "critic": critic_state,
            "entropy": entropy_state,
            "observation_normalizer": normalizer_state,
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
        model = DIME(config, train_env, eval_env, run_path, writer)
        target = {
            "actor": model.actor_state,
            "critic": model.critic_state,
            "entropy": model.entropy_state,
            "observation_normalizer": model.observation_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_directory, item=target, restore_args=restore_args)
        model.actor_state = checkpoint["actor"]
        model.target_actor_state = model.target_actor_state.replace(params=checkpoint["actor"].params)
        model.critic_state = checkpoint["critic"]
        model.entropy_state = checkpoint["entropy"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]
        shutil.rmtree(checkpoint_directory)
        return model


    def test(self, episodes):
        state, unused_info = self.eval_env.reset()
        completed_episodes = 0
        while completed_episodes < episodes:
            normalized_state = self.normalize(self.observation_normalizer_state, state)
            self.key, action, unused_running_cost, unused_stochastic_cost, unused_terminal_cost, unused_path = self.sample_action(self.actor_state.params, normalized_state, self.key, deterministic=True)
            if self.action_rescaling:
                action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
            state, unused_reward, terminated, truncated, unused_info = self.eval_env.step(jax.device_get(action))
            completed_episodes += int(np.sum(terminated | truncated))


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
