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
from rl_x.algorithms.ppo.flax_full_jit.actor_critic import ActorCritic
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

        def linear_schedule(count):
            frac = (
                1.0
                - (count // (self.nr_minibatches * self.nr_epochs))
                / self.nr_updates
            )
            return self.learning_rate * frac
        
        self.linear_schedule = linear_schedule

    
    def train(self):
        def _train(key):
            # INIT NETWORK
            policy, get_processed_action = get_policy(self.config, self.env)
            critic = get_critic(self.config, self.env)
            key, subkey1, subkey2 = jax.random.split(key, 3)
            init_x = jnp.zeros(self.env.single_observation_space.shape)
            network_params1 = policy.init(subkey1, init_x)
            network_params2 = critic.init(subkey2, init_x)
            if self.anneal_learning_rate:
                tx1 = optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
                )
                tx2 = optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
                )
            else:
                tx1 = optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(self.learning_rate, eps=1e-5),
                )
                tx2 = optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(self.learning_rate, eps=1e-5),
                )
            policy_state = TrainState.create(
                apply_fn=policy.apply,
                params=network_params1,
                tx=tx1,
            )
            critic_state = TrainState.create(
                apply_fn=critic.apply,
                params=network_params2,
                tx=tx2,
            )

            # INIT ENV
            key, subkey = jax.random.split(key)
            resetsubkey = jax.random.split(subkey, self.nr_envs)
            env_state = self.env.reset(resetsubkey)
            obsv = env_state.observation

            def _update_step(runner_state, unused):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, unused):
                    policy_state, critic_state, env_state, last_obs, key = runner_state

                    # SELECT ACTION
                    key, subkey = jax.random.split(key)
                    action_mean, action_logstd = policy.apply(policy_state.params, last_obs)
                    value = critic.apply(critic_state.params, last_obs)
                    value = value.squeeze(-1)
                    action_std = jnp.exp(action_logstd)
                    action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                    log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)

                    # STEP ENV
                    env_state = self.env.step(
                        env_state, action
                    )
                    obsv = env_state.observation
                    reward = env_state.reward
                    done = env_state.done
                    info = env_state.info
                    transition = Transition(
                        done, action, value, reward, log_prob, last_obs, info
                    )
                    runner_state = (policy_state, critic_state, env_state, obsv, key)
                    return runner_state, transition

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, self.nr_steps
                )

                # CALCULATE ADVANTAGE
                policy_state, critic_state, env_state, last_obs, key = runner_state
                last_val = critic.apply(critic_state.params, last_obs)
                last_val = last_val.squeeze(-1)

                def _calculate_gae(traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + self.gamma * next_value * (1 - done) - value
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * (1 - done) * gae
                        )
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj_batch.value

                advantages, targets = _calculate_gae(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        policy_state, critic_state = train_state
                        traj_batch, advantages, targets = batch_info

                        def _loss_fn(policy_params, critic_params, traj_batch, gae, targets):
                            # RERUN NETWORK
                            action_mean, action_logstd = policy.apply(policy_params, traj_batch.obs)
                            value = critic.apply(critic_params, traj_batch.obs)
                            action_std = jnp.exp(action_logstd)
                            log_prob = (-0.5 * ((traj_batch.action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                            entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                            entropy_loss = entropy.sum()
                            value = value.squeeze(-1)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-self.clip_range, self.clip_range)
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - self.clip_range,
                                    1.0 + self.clip_range,
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()

                            total_loss = (
                                loss_actor
                                + self.critic_coef * value_loss
                                - self.entropy_coef * entropy_loss
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                        total_loss, (policy_gradients, critic_gradients) = grad_fn(
                            policy_state.params, critic_state.params, traj_batch, advantages, targets
                        )
                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        return (policy_state, critic_state), total_loss

                    policy_state, critic_state, traj_batch, advantages, targets, key = update_state
                    key, subkey = jax.random.split(key)
                    batch_size = self.minibatch_size * self.nr_minibatches
                    assert (
                        batch_size == self.nr_steps * self.nr_envs
                    ), "batch size must be equal to number of steps * number of envs"
                    permutation = jax.random.permutation(subkey, batch_size)
                    batch = (traj_batch, advantages, targets)
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
                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, (policy_state, critic_state), minibatches
                    )
                    policy_state, critic_state = train_state
                    update_state = (policy_state, critic_state, traj_batch, advantages, targets, key)
                    return update_state, total_loss

                update_state = (policy_state, critic_state, traj_batch, advantages, targets, key)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, self.nr_epochs
                )
                policy_state = update_state[0]
                critic_state = update_state[1]
                metric = traj_batch.info
                key = update_state[-1]
                if True:

                    def callback(info):
                        print(info)

                    jax.debug.callback(callback, jnp.mean(metric["xy_vel_diff_norm"]))
                
                runner_state = (policy_state, critic_state, env_state, last_obs, key)
                return runner_state, None
                
            key, subkey = jax.random.split(key)
            runner_state = (policy_state, critic_state, env_state, obsv, subkey)
            jax.lax.scan(_update_step, runner_state, None, self.nr_updates)
            
        
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
