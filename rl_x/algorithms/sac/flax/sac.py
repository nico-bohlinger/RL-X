import os
import random
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from rl_x.algorithms.sac.flax.policy import get_policy
from rl_x.algorithms.sac.flax.critic import get_critic
from rl_x.algorithms.sac.flax.entropy_coefficient import EntropyCoefficient, ConstantEntropyCoefficient


class SAC():
    def __init__(self, config, env, writer) -> None:
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(config.runner.run_path, "models")
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.algorithm.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.q_update_freq = config.algorithm.q_update_freq
        self.q_update_steps = config.algorithm.q_update_steps
        self.q_target_update_freq = config.algorithm.q_target_update_freq
        self.policy_update_freq = config.algorithm.policy_update_freq
        self.policy_update_steps = config.algorithm.policy_update_steps
        self.entropy_update_freq = config.algorithm.entropy_update_freq
        self.entropy_update_steps = config.algorithm.entropy_update_steps
        self.entropy_coef = config.algorithm.entropy_coef
        self.target_entropy = config.algorithm.target_entropy
        self.log_freq = config.algorithm.log_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units

        if config.algorithm.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, alpha_key = jax.random.split(self.key, 4)

        self.policy, self.get_processed_action = get_policy(config, env)
        self.vector_critic = get_critic(config, env)
        
        if self.entropy_coef == "auto":
            if self.target_entropy == "auto":
                self.target_entropy = -np.prod(env.get_single_action_space_shape()).item()
            else:
                self.target_entropy = float(self.target_entropy)
            self.alpha = EntropyCoefficient(self.target_entropy)
        else:
            self.alpha = ConstantEntropyCoefficient(self.entropy_coef)

        def q_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps / self.q_update_freq * self.q_update_steps))
            return self.learning_rate * fraction
    
        def policy_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps / self.policy_update_freq * self.policy_update_steps))
            return self.learning_rate * fraction

        def entropy_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps / self.entropy_update_freq * self.entropy_update_steps))
            return self.entropy_coef * fraction
        
        q_learning_rate = q_linear_schedule if self.anneal_learning_rate else self.learning_rate
        policy_learning_rate = policy_linear_schedule if self.anneal_learning_rate else self.learning_rate
        entropy_learning_rate = entropy_linear_schedule if self.anneal_learning_rate else self.entropy_coef

        state = jnp.array([self.env.observation_space.sample()])
        action = jnp.array([self.env.action_space.sample()])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            tx=optax.adam(learning_rate=policy_learning_rate)
        )

        self.vector_critic_state = TrainState.create(
            apply_fn=self.vector_critic.apply,
            params=self.vector_critic.init(critic_key, state, action),
            tx=optax.adam(learning_rate=q_learning_rate)
        )

        self.alpha_state = TrainState.create(
            apply_fn=self.alpha.apply,
            params=self.alpha.init(alpha_key),
            tx=optax.adam(learning_rate=entropy_learning_rate)
        )

        self.policy.apply = jax.jit(self.policy.apply)
        self.vector_critic.apply = jax.jit(self.vector_critic.apply)
        self.alpha.apply = jax.jit(self.alpha.apply)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
        
    
    def train(self):
        exit()


    def test(self):
        pass
