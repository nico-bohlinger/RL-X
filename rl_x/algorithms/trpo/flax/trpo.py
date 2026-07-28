import os
import shutil
import json
import logging
import time
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.trpo.flax.general_properties import GeneralProperties
from rl_x.algorithms.trpo.flax.policy import get_policy
from rl_x.algorithms.trpo.flax.critic import get_critic
from rl_x.algorithms.trpo.flax.batch import Batch

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
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
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
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_critic_minibatches = self.batch_size // self.critic_minibatch_size
        self.nr_critic_optimizer_steps = self.nr_updates * self.nr_critic_updates * self.nr_critic_minibatches

        if self.batch_size % self.critic_minibatch_size != 0:
            raise ValueError("Rollout batch size must be divisible by critic minibatch size")
        if self.batch_size % self.policy_subsampling_factor != 0:
            raise ValueError("Rollout batch size must be divisible by policy subsampling factor")

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            return self.critic_learning_rate * (1.0 - count / self.nr_critic_optimizer_steps)

        critic_learning_rate = linear_schedule if self.anneal_critic_learning_rate else self.critic_learning_rate

        state = jnp.array([self.train_env.single_observation_space.sample()])

        self.policy_state = TrainState.create(apply_fn=self.policy.apply, params=self.policy.init(policy_key, state), tx=optax.set_to_zero())

        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=self.critic.init(critic_key, state), tx=optax.chain(optax.clip_by_global_norm(self.critic_max_grad_norm), optax.inject_hyperparams(optax.adam)(learning_rate=critic_learning_rate)))

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def train(self):
        @jax.jit
        def get_action_and_value(policy_state: TrainState, critic_state: TrainState, state: np.ndarray, key: jax.random.PRNGKey):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            action_std = jnp.exp(action_logstd)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            value = self.critic.apply(critic_state.params, state)
            processed_action = self.get_processed_action(action)
            return processed_action, action, value.reshape(-1), log_prob.sum(1), key


        @jax.jit
        def calculate_gae_advantages(critic_state, next_states, rewards, terminations, truncations, values):
            next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)

            def advantage_step(next_advantage, inputs):
                reward, value, next_value, terminated, truncated = inputs
                delta = reward + self.gamma * (1.0 - terminated) * next_value - value
                continuation = (1.0 - terminated) * (1.0 - truncated)
                advantage = delta + self.gamma * self.gae_lambda * continuation * next_advantage
                return advantage, advantage

            _, advantages = jax.lax.scan(advantage_step, jnp.zeros_like(values[-1]), (rewards, values, next_values, terminations, truncations), reverse=True)
            return advantages, advantages + values


        @jax.jit
        def update(policy_state, critic_state, states, actions, advantages, returns, values, log_probs, key):
            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = values.reshape(-1)
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
                new_log_prob = jnp.sum(-0.5 * ((policy_actions - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd, axis=-1)
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

            def conjugate_gradient_iteration(carry, unused):
                solution, residual, direction, residual_squared = carry
                hessian_direction = hessian_vector_product(direction)
                alpha = residual_squared / jnp.maximum(jnp.dot(direction, hessian_direction), 1e-8)
                next_solution = solution + alpha * direction
                next_residual = residual - alpha * hessian_direction
                next_residual_squared = jnp.dot(next_residual, next_residual)
                beta = next_residual_squared / jnp.maximum(residual_squared, 1e-8)
                active = residual_squared > self.cg_residual_tolerance
                return (jnp.where(active, next_solution, solution), jnp.where(active, next_residual, residual), jnp.where(active, next_residual + beta * direction, direction), jnp.where(active, next_residual_squared, residual_squared)), None

            residual_squared = jnp.dot(policy_gradient, policy_gradient)
            cg_carry = (jnp.zeros_like(policy_gradient), policy_gradient, policy_gradient, residual_squared)
            cg_carry, _ = jax.lax.scan(conjugate_gradient_iteration, cg_carry, None, self.cg_max_steps)
            search_direction, unused_residual, unused_direction, final_residual_squared = cg_carry
            search_curvature = jnp.dot(search_direction, hessian_vector_product(search_direction))
            full_step = jnp.sqrt(2.0 * self.target_kl / jnp.maximum(search_curvature, 1e-8)) * search_direction

            def line_search_iteration(carry, line_search_step):
                accepted_params, accepted, accepted_objective, accepted_kl, accepted_step = carry
                fraction = self.line_search_shrinking_factor ** line_search_step
                candidate_params = flat_policy_params + fraction * full_step
                candidate_objective = policy_objective(candidate_params)
                candidate_kl = mean_kl(candidate_params)
                valid = (~accepted) & jnp.isfinite(candidate_objective) & jnp.isfinite(candidate_kl) & (candidate_objective > old_policy_objective) & (candidate_kl <= self.target_kl)
                return (jnp.where(valid, candidate_params, accepted_params), accepted | valid, jnp.where(valid, candidate_objective, accepted_objective), jnp.where(valid, candidate_kl, accepted_kl), jnp.where(valid, line_search_step, accepted_step)), None

            line_search_carry = flat_policy_params, jnp.array(False), old_policy_objective, jnp.array(0.0), jnp.array(self.line_search_max_steps)
            line_search_carry, _ = jax.lax.scan(line_search_iteration, line_search_carry, jnp.arange(self.line_search_max_steps))
            accepted_params, line_search_success, new_policy_objective, new_kl, accepted_step = line_search_carry
            policy_state = policy_state.replace(params=unravel_policy_params(accepted_params))

            def critic_loss(critic_params, state_b, return_b):
                value = self.critic.apply(critic_params, state_b).squeeze(-1)
                return 0.5 * jnp.mean((value - return_b) ** 2)

            critic_grad_loss = jax.value_and_grad(critic_loss)
            key, subkey = jax.random.split(key)
            critic_batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_critic_updates, 1))
            critic_batch_indices = jax.random.permutation(subkey, critic_batch_indices, axis=1, independent=True).reshape((self.nr_critic_updates * self.nr_critic_minibatches, self.critic_minibatch_size))

            def critic_minibatch_update(state, minibatch_indices):
                loss, gradients = critic_grad_loss(state.params, batch_states[minibatch_indices], batch_returns[minibatch_indices])
                return state.apply_gradients(grads=gradients), (loss, optax.global_norm(gradients))

            critic_state, (critic_losses, critic_gradient_norms) = jax.lax.scan(critic_minibatch_update, critic_state, critic_batch_indices)
            metrics = {
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
                "v_value/explained_variance": 1.0 - jnp.var(batch_returns - batch_values) / (jnp.var(batch_returns) + 1e-8),
            }
            return policy_state, critic_state, metrics, key


        @jax.jit
        def get_deterministic_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)


        self.set_train_mode()

        batch = Batch(
            states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            next_states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=np.zeros((self.nr_steps, self.nr_envs) + self.as_shape),
            rewards=np.zeros((self.nr_steps, self.nr_envs)),
            values=np.zeros((self.nr_steps, self.nr_envs)),
            terminations=np.zeros((self.nr_steps, self.nr_envs)),
            truncations=np.zeros((self.nr_steps, self.nr_envs)),
            log_probs=np.zeros((self.nr_steps, self.nr_envs)),
            advantages=np.zeros((self.nr_steps, self.nr_envs)),
            returns=np.zeros((self.nr_steps, self.nr_envs)),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None

        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}
            if logging_time_prev:
                time_metrics["time/logging_time_prev"] = logging_time_prev


            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                processed_action, action, value, log_prob, self.key = get_action_and_value(self.policy_state, self.critic_state, state, self.key)
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

                batch.states[step] = state
                batch.next_states[step] = actual_next_state
                batch.actions[step] = action
                batch.rewards[step] = reward
                batch.values[step] = value
                batch.terminations[step] = terminated
                batch.truncations[step] = truncated
                batch.log_probs[step] = log_prob
                state = next_state
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating advantages and returns
            batch.advantages, batch.returns = calculate_gae_advantages(self.critic_state, batch.next_states, batch.rewards, batch.terminations, batch.truncations, batch.values)

            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            self.policy_state, self.critic_state, optimization_metrics, self.key = update(self.policy_state, self.critic_state, batch.states, batch.actions, batch.advantages, batch.returns, batch.values, batch.log_probs, self.key)
            optimization_metrics = {key: value.item() for key, value in optimization_metrics.items()}
            nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    eval_processed_action = get_deterministic_action(self.policy_state, eval_state)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(eval_processed_action))
                    eval_done = eval_terminated | eval_truncated
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics["eval/episode_return"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i))
                            evaluation_metrics["eval/episode_length"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                self.set_train_mode()

            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time


            # Saving
            # Also only save when there were finished episodes this update
            if self.save_model and dones_this_rollout > 0:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()

            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time


            # Logging
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

            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

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
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)

        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                processed_action = get_action(self.policy_state, state)
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
