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
                self.env.action_space(None).shape[0], activation="tanh"
            )
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(self.env.observation_space(None).shape)
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
            obsv, env_state = self.env.reset(reset_rng, None)

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
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, self.nr_envs)
                    obsv, env_state, reward, done, info = self.env.step(
                        rng_step, env_state, action, None
                    )
                    transition = Transition(
                        done, action, value, reward, log_prob, last_obs, info
                    )
                    runner_state = (train_state, env_state, obsv, rng)
                    return runner_state, transition

                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, self.nr_steps
                )
            
            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, env_state, obsv, _rng)
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, self.nr_updates
            )
            return {"runner_state": runner_state, "metrics": metric}
            
        
        # TODO: Do potentially multiple seeds here
        rng = jax.random.PRNGKey(self.seed)
        train_jit = jax.jit(_train)
        out = train_jit(rng)

        print("ok")
        exit()
    

    def test(self, episodes):
        # TODO:
        ...


    def general_properties():
        return GeneralProperties
