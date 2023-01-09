import os
import logging
import pickle
import random
import time
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import wandb

from rl_x.algorithms.mpo.flax.policy import get_policy
from rl_x.algorithms.mpo.flax.critic import get_critic
from rl_x.algorithms.mpo.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.mpo.flax.types import AgentParams, DualParams, TrainingState

rlx_logger = logging.getLogger("rl_x")


class MPO():
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
        self.agent_learning_rate = config.algorithm.agent_learning_rate
        self.dual_learning_rate = config.algorithm.dual_learning_rate
        self.anneal_agent_learning_rate = config.algorithm.anneal_agent_learning_rate
        self.anneal_dual_learning_rate = config.algorithm.anneal_dual_learning_rate
        self.init_log_temperature = config.algorithm.init_log_temperature
        self.init_log_alpha_mean = config.algorithm.init_log_alpha_mean
        self.init_log_alpha_stddev = config.algorithm.init_log_alpha_stddev
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.ensemble_size = config.algorithm.ensemble_size
        self.logging_freq = config.algorithm.logging_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.max_nr_batches_needed = 1

        if config.algorithm.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.env_as_low = env.action_space.low
        self.env_as_high = env.action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.vector_critic = get_critic(config, env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.vector_critic.apply = jax.jit(self.vector_critic.apply)

        def agent_linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / (total_steps))
            return self.agent_learning_rate * fraction
        
        def dual_linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / (total_steps))
            return self.dual_learning_rate * fraction

        agent_learning_rate = agent_linear_schedule if self.anneal_agent_learning_rate else self.agent_learning_rate
        dual_learning_rate = dual_linear_schedule if self.anneal_dual_learning_rate else self.dual_learning_rate

        self.agent_optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=agent_learning_rate)
        self.dual_optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=dual_learning_rate)

        state = jnp.array([self.env.observation_space.sample()])
        action = jnp.array([self.env.action_space.sample()])

        agent_params = AgentParams(
            policy_params=self.policy.init(policy_key, state),
            critic_params=self.vector_critic.init(critic_key, state, action)
        )
        
        dual_variable_shape = [np.prod(env.get_single_action_space_shape()).item()]

        dual_params = DualParams(
            log_temperature=jnp.full([1], self.init_log_temperature, dtype=jnp.float32),
            log_alpha_mean=jnp.full(dual_variable_shape, self.init_log_alpha_mean, dtype=jnp.float32),
            log_alpha_stddev=jnp.full(dual_variable_shape, self.init_log_alpha_stddev, dtype=jnp.float32)
        )

        self.train_state = TrainingState(
            agent_params=agent_params,
            agent_target_params=agent_params,
            dual_params=dual_params,
            agent_optimizer_state=self.agent_optimizer.init(agent_params),
            dual_optimizer_state=self.dual_optimizer.init(dual_params),
            steps=0,
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
        
    
    def train(self):
        @jax.jit
        def get_action(policy_params: flax.core.FrozenDict, state: np.ndarray, key: jax.random.PRNGKey):
            dist = self.policy.apply(policy_params, state)
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            return action, key


        @jax.jit
        def update():
            pass


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.observation_space.shape, self.env.action_space.shape)

        saving_return_buffer = deque(maxlen=100)
        episode_info_buffer = deque(maxlen=self.logging_freq)
        acting_time_buffer = deque(maxlen=self.logging_freq)
        optimize_time_buffer = deque(maxlen=self.logging_freq)
        saving_time_buffer = deque(maxlen=self.logging_freq)
        fps_buffer = deque(maxlen=self.logging_freq)

        state = self.env.reset()

        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                action, self.key = get_action(self.train_state.agent_params.policy_params, state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, done, info = self.env.step(jax.device_get(processed_action))
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    maybe_terminal_observation = self.env.get_terminal_observation(info, i)
                    if maybe_terminal_observation is not None:
                        actual_next_state[i] = maybe_terminal_observation
            
            replay_buffer.add(state, actual_next_state, action, reward, done)

            state = next_state
            global_step += self.nr_envs

            episode_infos = self.env.get_episode_infos(info)
            episode_info_buffer.extend(episode_infos)
            saving_return_buffer.extend([ep_info["r"] for ep_info in episode_infos])

            acting_end_time = time.time()
            acting_time_buffer.append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_try_to_save = should_learning_start and self.save_model and episode_infos
            should_log = global_step % self.logging_freq == 0


            # Optimizing - Prepare batches
            if should_optimize:
                ...
                # TODO


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                ...
                # TODO
            
            optimize_end_time = time.time()
            optimize_time_buffer.append(optimize_end_time - acting_end_time)


            # Saving
            if should_try_to_save:
                ...
                # TODO
            
            saving_end_time = time.time()
            saving_time_buffer.append(saving_end_time - optimize_end_time)

            fps_buffer.append(self.nr_envs / (saving_end_time - start_time))


            # Logging                
            if should_log:
                if self.track_console:
                    rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐")
                    self.log_console("global_step", global_step)
                else:
                    rlx_logger.info(f"Step: {global_step}")

                if len(episode_info_buffer) > 0:
                    self.log("rollout/ep_rew_mean", np.mean([ep_info["r"] for ep_info in episode_info_buffer]), global_step)
                    self.log("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in episode_info_buffer]), global_step)
                    names = list(episode_info_buffer[0].keys())
                    for name in names:
                        if name != "r" and name != "l" and name != "t":
                            self.log(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info.keys()]), global_step)
                self.log("time/fps", np.mean(fps_buffer), global_step)
                self.log("time/acting_time", np.mean(acting_time_buffer), global_step)
                self.log("time/optimize_time", np.mean(optimize_time_buffer), global_step)
                self.log("time/saving_time", np.mean(saving_time_buffer), global_step)
                # self.log("train/agent_learning_rate", self.train_state.agent_optimizer_state.hyperparams["learning_rate"].item(), global_step)
                # self.log("train/dual_learning_rate", self.train_state.dual_optimizer_state.hyperparams["learning_rate"].item(), global_step)

                if self.track_console:
                    rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

                episode_info_buffer.clear()
                acting_time_buffer.clear()
                optimize_time_buffer.clear()
                saving_time_buffer.clear()
                fps_buffer.clear()


    def get_buffer_mean(self, buffer):
        if len(buffer) > 0:
            return np.mean(buffer)
        else:
            return 0.0


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │")
        

    def save(self):
        # TODO
        pass
    
            
    def load(config, env, run_path, writer):
        # TODO
        pass
    

    def test(self, episodes):
        # TODO
        pass
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...
