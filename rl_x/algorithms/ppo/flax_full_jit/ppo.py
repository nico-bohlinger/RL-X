import os
import shutil
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

from rl_x.algorithms.ppo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.ppo.flax.policy import get_policy
from rl_x.algorithms.ppo.flax.critic import get_critic
from rl_x.algorithms.ppo.flax_full_jit.transition import Transition

rlx_logger = logging.getLogger("rl_x")


class PPO:
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
        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)

 
    def train(self):
        def jitable_train_function(key):
            key, policy_key, critic_key, state_key = jax.random.split(key, 4)

            policy, get_processed_action = get_policy(self.config, self.env)
            critic = get_critic(self.config, self.env)

            def linear_schedule(count):
                fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
                return self.learning_rate * fraction

            learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

            state = jnp.array([self.env.single_observation_space.sample(state_key)])

            policy_state = TrainState.create(
                apply_fn=policy.apply,
                params=policy.init(policy_key, state),
                tx=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
                )
            )

            critic_state = TrainState.create(
                apply_fn=critic.apply,
                params=critic.init(critic_key, state),
                tx=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
                )
            )

            key, subkey = jax.random.split(key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.env.reset(reset_keys)


            def learning_iteration(learning_iteration_carry, _):
                policy_state, critic_state, env_state, state, key = learning_iteration_carry

                # Acting
                def single_rollout(single_rollout_carry, _):
                    policy_state, critic_state, env_state, state, key = single_rollout_carry

                    key, subkey = jax.random.split(key)
                    action_mean, action_logstd = policy.apply(policy_state.params, state)
                    action_std = jnp.exp(action_logstd)
                    action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                    log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                    processed_action = get_processed_action(action)
                    value = critic.apply(critic_state.params, state).squeeze(-1)

                    env_state = self.env.step(env_state, processed_action)
                    next_state = env_state.next_observation
                    reward = env_state.reward
                    terminated = env_state.terminated
                    info = env_state.info
                    actual_next_state = env_state.actual_next_observation
                    transition = (state, actual_next_state, action, reward, value, terminated, log_prob, info)

                    return (policy_state, critic_state, env_state, next_state, key), transition

                single_rollout_carry, batch = jax.lax.scan(single_rollout, learning_iteration_carry, None, self.nr_steps)
                policy_state, critic_state, env_state, state, key = single_rollout_carry
                states, next_states, actions, rewards, values, terminations, log_probs, infos = batch


                # Calculating advantages and returns
                def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
                    def compute_advantages(carry, t):
                        prev_advantage = carry[0]
                        advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                        return (advantage,), advantage

                    next_values = critic.apply(critic_state.params, next_states).squeeze(-1)
                    delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                    init_advantages = delta[-1]
                    _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                    advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                    returns = advantages + values
                    return advantages, returns

                advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values, terminations)


                # Optimizing
                def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                    # Policy loss
                    action_mean, action_logstd = policy.apply(policy_params, state_b)
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
                    new_value = critic.apply(critic_params, state_b)
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


                # Evaluating
                ...



                # Saving
                ...


                # Logging
                def callback(infos):
                    self.start_logging(0)
                    for key, value in infos.items():
                        if key == "key":
                            continue
                        self.log(f"{key}", value, 0)
                    self.end_logging()

                infos = tree.map_structure(lambda x: jnp.mean(x), infos)
                jax.debug.callback(callback, infos)
                
                return (policy_state, critic_state, env_state, state, key), None
                
            key, subkey = jax.random.split(key)
            jax.lax.scan(learning_iteration, (policy_state, critic_state, env_state, env_state.next_observation, subkey), None, self.nr_updates)
            
        
        # TODO: Do potentially multiple seeds here
        key = jax.random.PRNGKey(self.seed)
        train_function = jax.jit(jitable_train_function)
        t0 = time.time()
        jax.block_until_ready(train_function(key))
        rlx_logger.info(f"time: {time.time() - t0:.2f} s")
    

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


    def test(self, episodes):
        # TODO:
        ...


    def general_properties():
        return GeneralProperties
