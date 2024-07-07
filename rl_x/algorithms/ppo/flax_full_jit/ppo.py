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

        def linear_schedule(count):
            frac = (
                1.0
                - (count // (self.nr_minibatches * self.nr_epochs))
                / self.nr_updates
            )
            return self.learning_rate * frac
        
        self.linear_schedule = linear_schedule

    
    def train(self):
        def _train(rng):
            # INIT NETWORK
            network = ActorCritic(
                action_dim=8, activation="tanh"
            )
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(27)
            network_params = network.init(_rng, init_x)
            if self.anneal_learning_rate:
                tx = optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(self.learning_rate, eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, self.nr_envs)
            env_state = self.env.reset(reset_rng)
            obsv = env_state.observation

            def _update_step(runner_state, unused):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, unused):
                    train_state, env_state, last_obs, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    pi, value = network.apply(train_state.params, last_obs)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

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
                    runner_state = (train_state, env_state, obsv, rng)
                    return runner_state, transition

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, self.nr_steps
                )

                # CALCULATE ADVANTAGE
                train_state, env_state, last_obs, rng = runner_state
                _, last_val = network.apply(train_state.params, last_obs)

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
                        traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, traj_batch, gae, targets):
                            # RERUN NETWORK
                            pi, value = network.apply(params, traj_batch.obs)
                            log_prob = pi.log_prob(traj_batch.action)

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
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + self.critic_coef * value_loss
                                - self.entropy_coef * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    train_state, traj_batch, advantages, targets, rng = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = self.minibatch_size * self.nr_minibatches
                    assert (
                        batch_size == self.nr_steps * self.nr_envs
                    ), "batch size must be equal to number of steps * number of envs"
                    permutation = jax.random.permutation(_rng, batch_size)
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
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (train_state, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, self.nr_epochs
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
                if False:

                    def callback(info):
                        print("update done")

                    jax.debug.callback(callback, metric)
                
                runner_state = (train_state, env_state, last_obs, rng)
                return runner_state, metric
                
            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, env_state, obsv, _rng)
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, self.nr_updates
            )
            return {"runner_state": runner_state, "metrics": metric}
            
        
        # TODO: Do potentially multiple seeds here

        import time
        import matplotlib.pyplot as plt
        rng = jax.random.PRNGKey(self.seed)
        train_jit = jax.jit(_train)
        print("Ready")
        t0 = time.time()
        out = jax.block_until_ready(train_jit(rng))
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
