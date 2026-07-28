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
from jax.flatten_util import ravel_pytree
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.trpo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.trpo.flax_full_jit.policy import get_policy
from rl_x.algorithms.trpo.flax_full_jit.critic import get_critic

rlx_logger = logging.getLogger("rl_x")


class TRPO:
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
        self.render_callback_type = getattr(config.environment, 'render_callback_type', 'io_callback')
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.anneal_critic_learning_rate = config.algorithm.anneal_critic_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.critic_minibatch_size = config.algorithm.critic_minibatch_size
        self.nr_critic_updates = config.algorithm.nr_critic_updates
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.target_kl = config.algorithm.target_kl
        self.cg_max_steps = config.algorithm.cg_max_steps
        self.cg_damping = config.algorithm.cg_damping
        self.cg_residual_tolerance = config.algorithm.cg_residual_tolerance
        self.line_search_shrinking_factor = config.algorithm.line_search_shrinking_factor
        self.line_search_max_steps = config.algorithm.line_search_max_steps
        self.policy_subsampling_factor = config.algorithm.policy_subsampling_factor
        self.critic_max_grad_norm = config.algorithm.critic_max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_critic_minibatches = self.batch_size // self.critic_minibatch_size
        self.nr_critic_optimizer_steps = self.nr_updates * self.nr_critic_updates * self.nr_critic_minibatches
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon

        if self.total_timesteps % self.batch_size != 0:
            raise ValueError("Total timesteps must be divisible by the rollout batch size")
        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")
        if self.batch_size % self.critic_minibatch_size != 0:
            raise ValueError("Rollout batch size must be divisible by critic minibatch size")
        if self.batch_size % self.policy_subsampling_factor != 0:
            raise ValueError("Rollout batch size must be divisible by policy subsampling factor")
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time.")
        if self.target_kl <= 0:
            raise ValueError("Target KL must be positive")
        if self.cg_max_steps < 1:
            raise ValueError("Conjugate gradient steps must be at least one")
        if self.cg_damping < 0:
            raise ValueError("Conjugate gradient damping must be non-negative")
        if not 0 < self.line_search_shrinking_factor < 1:
            raise ValueError("Line search shrinking factor must be between zero and one")
        if self.line_search_max_steps < 1:
            raise ValueError("Line search steps must be at least one")
        if self.policy_subsampling_factor < 1:
            raise ValueError("Policy subsampling factor must be at least one")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - count / self.nr_critic_optimizer_steps
            return self.critic_learning_rate * fraction

        critic_learning_rate = linear_schedule if self.anneal_critic_learning_rate else self.critic_learning_rate
        env_state = self.train_env.reset(reset_key, False)

        self.policy_state = TrainState.create(apply_fn=self.policy.apply, params=self.policy.init(policy_key, env_state.next_observation), tx=optax.set_to_zero())

        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=self.critic.init(critic_key, env_state.next_observation), tx=optax.chain(optax.clip_by_global_norm(self.critic_max_grad_norm), optax.inject_hyperparams(optax.adam)(learning_rate=critic_learning_rate)))

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            policy_state = self.policy_state
            critic_state = self.critic_state

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, critic_state, env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, env_state, key = learning_iteration_carry

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, env_state, key = single_rollout_carry

                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(-1)
                        processed_action = self.get_processed_action(action)
                        value = self.critic.apply(critic_state.params, observation).squeeze(-1)

                        env_state = self.train_env.step(env_state, processed_action)
                        transition = (observation, env_state.actual_next_observation, action, env_state.reward, value, env_state.terminated, log_prob, env_state.info)

                        if self.render:
                            if self.render_callback_type == "debug_callback":
                                jax.debug.callback(self.train_env.render, env_state)
                            else:
                                def render(env_state):
                                    return self.train_env.render(env_state)
                                env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, learning_iteration_carry, None, self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, log_probs, infos = batch

                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1), unroll=True)
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                    advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values, terminations)

                    # Optimizing
                    # Policy update
                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)
                    normalized_advantages = (batch_advantages - jnp.mean(batch_advantages)) / (jnp.std(batch_advantages) + 1e-8)

                    policy_states = batch_states[::self.policy_subsampling_factor]
                    policy_actions = batch_actions[::self.policy_subsampling_factor]
                    policy_advantages = normalized_advantages[::self.policy_subsampling_factor]
                    policy_old_log_probs = batch_log_probs[::self.policy_subsampling_factor]
                    old_action_mean, old_action_logstd = self.policy.apply(policy_state.params, policy_states)
                    old_action_mean = jax.lax.stop_gradient(old_action_mean)
                    old_action_logstd = jax.lax.stop_gradient(old_action_logstd)
                    flat_policy_params, unravel_policy_params = ravel_pytree(policy_state.params)

                    def policy_objective(flat_params):
                        action_mean, action_logstd = self.policy.apply(unravel_policy_params(flat_params), policy_states)
                        action_std = jnp.exp(action_logstd)
                        new_log_prob = (-0.5 * ((policy_actions - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(-1)
                        return jnp.mean(policy_advantages * jnp.exp(new_log_prob - policy_old_log_probs))

                    def mean_kl(flat_params):
                        action_mean, action_logstd = self.policy.apply(unravel_policy_params(flat_params), policy_states)
                        old_variance = jnp.exp(2.0 * old_action_logstd)
                        new_variance = jnp.exp(2.0 * action_logstd)
                        kl = action_logstd - old_action_logstd + (old_variance + (old_action_mean - action_mean) ** 2) / (2.0 * new_variance) - 0.5
                        return jnp.mean(jnp.sum(kl, axis=-1))

                    old_policy_objective, policy_gradient = jax.value_and_grad(policy_objective)(flat_policy_params)
                    kl_gradient = jax.grad(mean_kl)

                    def hessian_vector_product(vector):
                        return jax.jvp(kl_gradient, (flat_policy_params,), (vector,))[1] + self.cg_damping * vector

                    def conjugate_gradient_iteration(carry, _):
                        solution, residual, direction, residual_squared = carry
                        hessian_direction = hessian_vector_product(direction)
                        alpha = residual_squared / jnp.maximum(jnp.dot(direction, hessian_direction), 1e-8)
                        next_solution = solution + alpha * direction
                        next_residual = residual - alpha * hessian_direction
                        next_residual_squared = jnp.dot(next_residual, next_residual)
                        beta = next_residual_squared / jnp.maximum(residual_squared, 1e-8)
                        next_direction = next_residual + beta * direction
                        active = residual_squared > self.cg_residual_tolerance
                        solution = jnp.where(active, next_solution, solution)
                        residual = jnp.where(active, next_residual, residual)
                        direction = jnp.where(active, next_direction, direction)
                        residual_squared = jnp.where(active, next_residual_squared, residual_squared)
                        return (solution, residual, direction, residual_squared), None

                    initial_residual_squared = jnp.dot(policy_gradient, policy_gradient)
                    conjugate_gradient_carry = (jnp.zeros_like(policy_gradient), policy_gradient, policy_gradient, initial_residual_squared)
                    conjugate_gradient_carry, _ = jax.lax.scan(conjugate_gradient_iteration, conjugate_gradient_carry, None, self.cg_max_steps)
                    search_direction, _, _, final_residual_squared = conjugate_gradient_carry
                    search_curvature = jnp.dot(search_direction, hessian_vector_product(search_direction))
                    maximum_step_size = jnp.sqrt(2.0 * self.target_kl / jnp.maximum(search_curvature, 1e-8))
                    full_step = maximum_step_size * search_direction

                    def line_search_iteration(carry, line_search_step):
                        accepted_params, accepted, accepted_objective, accepted_kl, accepted_step = carry
                        fraction = self.line_search_shrinking_factor ** line_search_step
                        candidate_params = flat_policy_params + fraction * full_step
                        candidate_objective = policy_objective(candidate_params)
                        candidate_kl = mean_kl(candidate_params)
                        valid = jnp.logical_and(jnp.logical_not(accepted), jnp.isfinite(candidate_objective))
                        valid = jnp.logical_and(valid, jnp.isfinite(candidate_kl))
                        valid = jnp.logical_and(valid, candidate_objective > old_policy_objective)
                        valid = jnp.logical_and(valid, candidate_kl <= self.target_kl)
                        accepted_params = jnp.where(valid, candidate_params, accepted_params)
                        accepted_objective = jnp.where(valid, candidate_objective, accepted_objective)
                        accepted_kl = jnp.where(valid, candidate_kl, accepted_kl)
                        accepted_step = jnp.where(valid, line_search_step, accepted_step)
                        accepted = jnp.logical_or(accepted, valid)
                        return (accepted_params, accepted, accepted_objective, accepted_kl, accepted_step), None

                    line_search_carry = (flat_policy_params, jnp.array(False), old_policy_objective, jnp.array(0.0), jnp.array(self.line_search_max_steps))
                    line_search_carry, _ = jax.lax.scan(line_search_iteration, line_search_carry, jnp.arange(self.line_search_max_steps))
                    accepted_params, line_search_success, new_policy_objective, new_kl, accepted_step = line_search_carry
                    policy_state = policy_state.replace(params=unravel_policy_params(accepted_params))

                    # Critic update
                    def critic_loss(critic_params, state_b, return_b):
                        value = self.critic.apply(critic_params, state_b).squeeze(-1)
                        return 0.5 * jnp.mean((value - return_b) ** 2)

                    critic_grad_loss = jax.value_and_grad(critic_loss)
                    key, subkey = jax.random.split(key)
                    critic_batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_critic_updates, 1))
                    critic_batch_indices = jax.random.permutation(subkey, critic_batch_indices, axis=1, independent=True)
                    critic_batch_indices = critic_batch_indices.reshape((self.nr_critic_updates * self.nr_critic_minibatches, self.critic_minibatch_size))

                    def critic_minibatch_update(critic_state, minibatch_indices):
                        critic_loss_value, critic_gradients = critic_grad_loss(critic_state.params, batch_states[minibatch_indices], batch_returns[minibatch_indices])
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        return critic_state, (critic_loss_value, optax.global_norm(critic_gradients))

                    critic_state, (critic_losses, critic_gradient_norms) = jax.lax.scan(critic_minibatch_update, critic_state, critic_batch_indices)

                    # Create metrics
                    optimization_metrics = {
                        "loss/policy_objective": new_policy_objective,
                        "loss/critic_loss": jnp.mean(critic_losses),
                        "policy/kl_divergence": new_kl,
                        "policy/line_search_success": line_search_success.astype(jnp.float32),
                        "policy/line_search_steps": accepted_step,
                        "policy/line_search_fraction": jnp.where(line_search_success, self.line_search_shrinking_factor ** accepted_step, 0.0),
                        "policy/objective_improvement": new_policy_objective - old_policy_objective,
                        "policy/expected_improvement": jnp.dot(policy_gradient, full_step),
                        "policy/std_dev": jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"])),
                        "conjugate_gradient/residual_norm": jnp.sqrt(final_residual_squared),
                        "conjugate_gradient/search_direction_norm": jnp.linalg.norm(search_direction),
                        "gradients/policy_grad_norm": jnp.linalg.norm(policy_gradient),
                        "gradients/critic_grad_norm": jnp.mean(critic_gradient_norms),
                        "lr/critic_learning_rate": critic_state.opt_state[1].hyperparams["learning_rate"],
                        "v_value/explained_variance": 1 - jnp.var(batch_returns - values.reshape(-1)) / (jnp.var(batch_returns) + 1e-8),
                    }

                    # Logging
                    combined_metrics = {**infos, **optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_policy_updates"] = combined_learning_iteration_step.item()
                        metrics["steps/nr_critic_updates"] = combined_learning_iteration_step.item() * self.nr_critic_updates * self.nr_critic_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id))

                    return (policy_state, critic_state, env_state, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, env_state, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, env_state, key = learning_iteration_carry

                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, eval_env_state = single_eval_rollout_carry

                        eval_action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = eval_action_mean
                        eval_processed_action = self.get_processed_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_processed_action)

                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state), jnp.arange(self.horizon))
                    _, eval_env_state = single_eval_rollout_carry

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    def callback(metrics_and_global_step):
                        metrics, combined_learning_iteration_step = metrics_and_global_step
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    combined_learning_iteration_step = (multi_learning_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration
                    jax.debug.callback(callback, (eval_metrics, combined_learning_iteration_step))

                # Saving
                if self.save_model:
                    def save_with_check(policy_state, critic_state):
                        self.save(policy_state, critic_state)
                    jax.debug.callback(save_with_check, policy_state, critic_state)

                return (policy_state, critic_state, env_state, key), None

            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, env_state, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")


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


    def save(self, policy_state, critic_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


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
        model = TRPO(config, train_env, eval_env, run_path, writer)

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
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            # key, subkey = jax.random.split(key)
            action_mean, action_logstd = self.policy.apply(self.policy_state.params, env_state.next_observation)
            # action_std = jnp.exp(action_logstd)
            action = action_mean # + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            processed_action = self.get_processed_action(action)
            env_state = self.train_env.step(env_state, processed_action)
            return env_state, key

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.train_env.reset(reset_keys, True)
        while True:
            env_state, self.key = rollout(env_state, self.key)
            if self.render:
                env_state = self.train_env.render(env_state)


    def general_properties():
        return GeneralProperties
