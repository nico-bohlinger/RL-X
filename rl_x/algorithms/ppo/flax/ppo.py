import os
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.ppo.flax.general_properties import GeneralProperties
from rl_x.algorithms.ppo.flax.policy import get_policy
from rl_x.algorithms.ppo.flax.critic import get_critic

rlx_logger = logging.getLogger("rl_x")


class PPO:
    def __init__(self, config, env, run_path, writer) -> None:
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
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        self.nr_rollouts = self.total_timesteps // (self.nr_steps * self.nr_envs)

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape
        
        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([env.single_observation_space.sample()])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "model_best_jax"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)

    
    def train(self):
        def train_loop():
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
            def calculate_gae_advantages(critic_state: TrainState, next_states: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, values: np.ndarray):
                def compute_advantages(carry, t):
                    prev_advantage = carry[0]
                    advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                    return (advantage,), advantage

                next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                init_advantages = delta[-1]
                _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                returns = advantages + values
                return advantages, returns
            

            @jax.jit
            def update(policy_state: TrainState, critic_state: TrainState,
                    states: np.ndarray, actions: np.ndarray, advantages: np.ndarray, returns: np.ndarray, values: np.ndarray, log_probs: np.ndarray,
                    key: jax.random.PRNGKey):
                def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                    # Policy loss
                    action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                    action_std = jnp.exp(action_logstd)
                    new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                    new_log_prob = new_log_prob.sum(1)
                    entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                    
                    logratio = new_log_prob - log_prob_b
                    ratio = jnp.exp(logratio)
                    approx_kl_div = (ratio - 1) - logratio
                    clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                    pg_loss1 = -advantage_b * ratio
                    pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = jnp.maximum(pg_loss1, pg_loss2)
                    
                    entropy_loss = entropy.sum(1)
                    
                    # Critic loss
                    new_value = self.critic.apply(critic_params, state_b)
                    critic_loss = 0.5 * (new_value - return_b) ** 2

                    # Combine losses
                    loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

                    # Create metrics
                    metrics = {
                        "loss/policy_gradient_loss": pg_loss,
                        "loss/critic_loss": critic_loss,
                        "loss/entropy_loss": entropy_loss,
                        "policy_ratio/approx_kl": approx_kl_div,
                        "policy_ratio/clip_fraction": clip_fraction,
                    }

                    return loss, (metrics)
                

                batch_states = states.reshape((-1,) + self.os_shape)
                batch_actions = actions.reshape((-1,) + self.as_shape)
                batch_advantages = advantages.reshape(-1)
                batch_returns = returns.reshape(-1)
                batch_log_probs = log_probs.reshape(-1)

                vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
                safe_mean = lambda x: jnp.mean(x) if x is not None else x
                mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                key, subkey = jax.random.split(key)
                batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                def minibatch_update(carry, minibatch_indices):
                    policy_state, critic_state = carry

                    minibatch_advantages = batch_advantages[minibatch_indices]
                    minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                    (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                        policy_state.params,
                        critic_state.params,
                        batch_states[minibatch_indices],
                        batch_actions[minibatch_indices],
                        batch_log_probs[minibatch_indices],
                        batch_returns[minibatch_indices],
                        minibatch_advantages
                    )

                    policy_state = policy_state.apply_gradients(grads=policy_gradients)
                    critic_state = critic_state.apply_gradients(grads=critic_gradients)

                    metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                    metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                    carry = (policy_state, critic_state)

                    return carry, (metrics)
                
                init_carry = (policy_state, critic_state)
                carry, (metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                policy_state, critic_state = carry

                # Calculate mean metrics
                mean_metrics = {key: jnp.mean(metrics[key]) for key in metrics}
                mean_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                mean_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                mean_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))

                return policy_state, critic_state, mean_metrics, key


            @jax.jit
            def get_deterministic_action(policy_state: TrainState, state: np.ndarray):
                action_mean, action_logstd = self.policy.apply(policy_state.params, state)
                return self.get_processed_action(action_mean)
            

            def env_reset_callback():
                state, _ = self.env.reset()
                state = state.astype(np.float32)
                return state
            

            def reset_step_info_collection_callback():
                self.step_info_collection = {}
            

            def env_step_callback(action):
                next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(action))
                next_state = next_state.astype(np.float32)
                reward = reward.astype(np.float32)

                actual_next_state = next_state.copy()
                done = terminated | truncated
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = self.env.get_final_observation_at_index(info, i)
                        self.saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                self.current_nr_episodes += np.sum(done)

                for key, info_value in self.env.get_logging_info_dict(info).items():
                    self.step_info_collection.setdefault(key, []).extend(info_value)

                return next_state, actual_next_state, reward, terminated, truncated
            

            def evaluate_callback(policy_state, old_state):
                self.evaluation_metrics = {}

                if self.evaluation_frequency != -1 and self.global_step % self.evaluation_frequency == 0:
                    self.set_eval_mode()
                    state, _ = self.env.reset()
                    eval_nr_episodes = 0
                    evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                    while True:
                        processed_action = get_deterministic_action(policy_state, state)
                        state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                        done = terminated | truncated
                        for i, single_done in enumerate(done):
                            if single_done:
                                eval_nr_episodes += 1
                                evaluation_metrics["eval/episode_return"].append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                                evaluation_metrics["eval/episode_length"].append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                                if eval_nr_episodes == self.evaluation_episodes:
                                    break
                        if eval_nr_episodes == self.evaluation_episodes:
                            break
                    self.evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                    state, _ = self.env.reset()
                    self.set_train_mode()
                    return state.astype(np.float32)
                else:
                    return old_state
            

            def save_callback():
                if self.save_model:
                    mean_return = np.mean(self.saving_return_buffer)
                    if mean_return > self.best_mean_return:
                        self.best_mean_return = mean_return
                        self.save()


            def log_callback(optimization_metrics):
                current_time = time.time()
                sps = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time))
                self.last_time = current_time
                time_metrics = {
                    "time/sps": sps
                }

                self.global_step += self.nr_steps * self.nr_envs
                self.current_nr_updates += self.nr_epochs * self.nr_minibatches
                steps_metrics = {
                    "steps/nr_env_steps": self.global_step,
                    "steps/nr_updates": self.current_nr_updates,
                    "steps/nr_episodes": self.current_nr_episodes
                }

                rollout_info_metrics = {}
                env_info_metrics = {}
                info_names = list(self.step_info_collection.keys())
                for info_name in info_names:
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(self.step_info_collection[info_name])
                    if mean_value == mean_value:  # Check if mean_value is NaN
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value

                additional_metrics = {**rollout_info_metrics, **self.evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics}

                self.start_logging(self.global_step)
                for key, value in additional_metrics.items():
                    self.log(key, value, self.global_step)
                for key, value in optimization_metrics.items():
                    self.log(key, value, self.global_step)
                self.end_logging()
            

            self.set_train_mode()

            state_shape = jax.ShapeDtypeStruct(self.np_state_shape, jnp.float32)
            reward_shape = jax.ShapeDtypeStruct((self.nr_envs,), jnp.float32)
            terminated_shape = jax.ShapeDtypeStruct((self.nr_envs,), jnp.bool_)
            truncated_shape = jax.ShapeDtypeStruct((self.nr_envs,), jnp.bool_)
            combined_callback_shapes = (state_shape, state_shape, reward_shape, terminated_shape, truncated_shape)

            state = jax.pure_callback(env_reset_callback, state_shape)

            def train(carry, _):
                def rollout(carry, _):
                    policy_state, critic_state, state, key = carry
                    processed_action, action, value, log_prob, key = get_action_and_value(policy_state, critic_state, state, key)
                    next_state, actual_next_state, reward, terminated, truncated = jax.pure_callback(env_step_callback, combined_callback_shapes, processed_action)

                    batch = (state, actual_next_state, action, reward, value, terminated, log_prob)

                    return (policy_state, critic_state, next_state, key), batch
                

                # Acting
                jax.debug.callback(reset_step_info_collection_callback)
                new_carry, batch = jax.lax.scan(rollout, carry, jnp.arange(self.nr_steps))


                # Calculating advantages and returns
                policy_state, critic_state, state, key = new_carry
                states, actual_next_states, actions, rewards, values, terminations, log_probs = batch
                advantages, returns = calculate_gae_advantages(critic_state, actual_next_states, rewards, terminations, values)


                # Optimizing
                policy_state, critic_state, optimization_metrics, key = update(
                    policy_state, critic_state,
                    states, actions, advantages, returns, values, log_probs,
                    key
                )


                # Evaluating
                state = jax.pure_callback(evaluate_callback, state_shape, policy_state, state)


                # Saving
                jax.debug.callback(save_callback)


                # Logging
                jax.debug.callback(log_callback, optimization_metrics)


                return (policy_state, critic_state, state, key), ()


            init_carry = (self.policy_state, self.critic_state, state, self.key)
            _, _ = jax.lax.scan(train, init_carry, jnp.arange(self.nr_rollouts))


        state, _ = self.env.reset()
        self.np_state_shape = state.shape
        self.saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        self.evaluation_metrics = {}
        self.last_time = time.time()
        self.global_step = 0
        self.current_nr_updates = 0
        self.current_nr_episodes = 0

        jitted_train_loop = jax.jit(train_loop)
        jitted_train_loop()


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
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": self.policy_state,
            "critic": self.critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        os.rename(f"{self.save_path}/tmp/{self.best_model_file_name}", f"{self.save_path}/{self.best_model_file_name}")
        os.remove(f"{self.save_path}/tmp/_METADATA")
        os.rmdir(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_file_name = splitted_path[-1]

        check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=checkpoint_file_name)
        checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)

        loaded_algorithm_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = PPO(config, env, run_path, writer)

        target = {
            "config_algorithm": config.algorithm.to_dict(),
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

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
