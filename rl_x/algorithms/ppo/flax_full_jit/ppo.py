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
        def _train(key):

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
                def _update_epoch(update_state, unused):
                    def _update_minbatch(policy_and_critic_states, batch_info):
                        policy_state, critic_state = policy_and_critic_states
                        states, actions, advantages, returns, values, log_probs = batch_info

                        def _loss_fn(policy_params, critic_params, states, actions, advantages, returns, values, log_probs):
                            # RERUN NETWORK
                            action_mean, action_logstd = policy.apply(policy_params, states)
                            action_std = jnp.exp(action_logstd)
                            log_prob = (-0.5 * ((actions - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                            entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                            entropy_loss = entropy.sum(1).mean()
                            value = critic.apply(critic_params, states)
                            value = value.squeeze(-1)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = values + (
                                value - values
                            ).clip(-self.clip_range, self.clip_range)
                            value_losses = jnp.square(value - returns)
                            value_losses_clipped = jnp.square(value_pred_clipped - returns)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - log_probs)
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                            loss_actor1 = ratio * advantages
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - self.clip_range,
                                    1.0 + self.clip_range,
                                )
                                * advantages
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()

                            total_loss = (
                                loss_actor
                                + self.critic_coef * value_loss
                                - self.entropy_coef * entropy_loss
                            )
                            return total_loss, (value_loss, loss_actor, entropy_loss)

                        grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                        total_loss, (policy_gradients, critic_gradients) = grad_fn(policy_state.params, critic_state.params, states, actions, advantages, returns, values, log_probs)
                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        return (policy_state, critic_state), total_loss

                    policy_state, critic_state, states, actions, advantages, returns, values, log_probs, key = update_state
                    key, subkey = jax.random.split(key)
                    batch_size = self.minibatch_size * self.nr_minibatches
                    assert (
                        batch_size == self.nr_steps * self.nr_envs
                    ), "batch size must be equal to number of steps * number of envs"
                    permutation = jax.random.permutation(subkey, batch_size)
                    batch = (states, actions, advantages, returns, values, log_probs)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x, [self.nr_minibatches, -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )
                    policy_and_critic_states, total_loss = jax.lax.scan(_update_minbatch, (policy_state, critic_state), minibatches)
                    policy_state, critic_state = policy_and_critic_states
                    update_state = (policy_state, critic_state, states, actions, advantages, returns, values, log_probs, key)
                    return update_state, total_loss

                update_state = (policy_state, critic_state, states, actions, advantages, returns, values, log_probs, key)
                update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, self.nr_epochs)
                policy_state = update_state[0]
                critic_state = update_state[1]
                metric = infos
                key = update_state[-1]
                if True:

                    def callback(info):
                        print(info)

                    jax.debug.callback(callback, jnp.mean(metric["xy_vel_diff_norm"]))
                
                return (policy_state, critic_state, env_state, state, key), None
                
            key, subkey = jax.random.split(key)
            jax.lax.scan(learning_iteration, (policy_state, critic_state, env_state, env_state.next_observation, subkey), None, self.nr_updates)
            
        
        # TODO: Do potentially multiple seeds here

        import time
        import matplotlib.pyplot as plt
        key = jax.random.PRNGKey(self.seed)
        train_jit = jax.jit(_train)
        print("Ready")
        t0 = time.time()
        jax.block_until_ready(train_jit(key))
        print(f"time: {time.time() - t0:.2f} s")
        # plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
        # plt.xlabel("Update Step")
        # plt.ylabel("Return")
        # plt.show()
    

    def test(self, episodes):
        # TODO:
        ...


    def general_properties():
        return GeneralProperties
