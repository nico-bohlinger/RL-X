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
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb
import re

from rl_x.algorithms.trirl_dtrl.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.trirl_dtrl.flax_full_jit.policy import get_policy
from rl_x.algorithms.trirl_dtrl.flax_full_jit.critic import get_critic
from rl_x.algorithms.trirl_dtrl.flax_full_jit.buffer import ParamsBuffer, EtasBuffer
from rl_x.algorithms.trirl_dtrl.flax.discriminator import get_discriminator, get_reward_approximator
from rl_x.algorithms.trirl_dtrl.data_utils import prepare_expert_data, expert_data_spec
from rl_x.algorithms.trirl_dtrl.flax_full_jit.reward_correction import make_chunked_ensemble_rew_correct
from rl_x.algorithms.trirl_dtrl.flax_full_jit.trust_region_layer import *
rlx_logger = logging.getLogger("rl_x")


class TRIRL_DTRL:
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
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon
        self.dim = np.prod(self.as_shape).item()

        # DTRL Projection layer and TRIRL attributes
        self.mean_bound = config.algorithm.mean_bound
        self.cov_bound = config.algorithm.cov_bound
        self.trust_region_coef = config.algorithm.trust_region_coef
        self.data_path = config.algorithm.data_path
        self.nr_epochs_disc = config.algorithm.nr_epochs_disc
        self.learning_rate_disc = config.algorithm.learning_rate_disc
        self.env_reward_frac = config.algorithm.env_reward_frac
        self.handle_absorbing_states = config.algorithm.handle_absorbing_states
        self.epsilon = config.algorithm.epsilon
        self.disc_buffer_capacity = config.algorithm.disc_buffer_capacity
        self.gp_lambda = config.algorithm.gp_lambda
        self.gp_alpha = config.algorithm.gp_alpha
        self.beta = 1/config.algorithm.entropy_coef
        self.num_data_samples = np.load(self.data_path)["states"].shape[0]

        chunk_size_dict = {10:30, 20:14, 50:8, 100:4} # dict mapping nr_steps to chunk size
        self.on_demand_etas = config.algorithm.on_demand_etas
        if self.on_demand_etas:
            chunk_size_dict = {k: int(v/2 - 1) for k, v in chunk_size_dict.items()}
            chunk_size_dict[10] = 20
            self.maximum_eta = False # always use per state eta if computing etas on demand
        else:
            self.maximum_eta = True
        self.chunk_size = 10
        self.reward_fn_approximator = config.algorithm.reward_fn_approximator
        self.nr_epochs_rew = config.algorithm.nr_epochs_rew
        self.learning_rate_reward_fn = config.algorithm.learning_rate_reward_fn

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")
        
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, discriminator_key, reset_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)
        self.discriminator = get_discriminator(config, self.train_env)
        self.reward_fn = get_reward_approximator(config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        def linear_schedule_disc(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs_disc)) / ((self.nr_updates * self.nr_epochs) / self.nr_epochs_disc)
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        learning_rate_disc = linear_schedule_disc if self.anneal_learning_rate else self.learning_rate_disc

        self.key, sampling_key = jax.random.split(self.key)
        env_state = self.train_env.reset(reset_key, False)
        action = jnp.array([self.train_env.single_action_space.sample(sampling_key)])
        self.H_terminal = jnp.sum(jnp.log(self.train_env.single_action_space.high - self.train_env.single_action_space.low))

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.discriminator_state = TrainState.create(
            apply_fn=self.discriminator.apply,
            params=self.discriminator.init(discriminator_key, env_state.next_observation, action, env_state.actual_next_observation, env_state.terminated),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate_disc),
            )
        )

        learning_rate_reward_fn = self.learning_rate_reward_fn
        self.key, reward_fn_key = jax.random.split(self.key)
        self.reward_fn_state = TrainState.create(
            apply_fn=self.reward_fn.apply,
            params=self.reward_fn.init(reward_fn_key, env_state.next_observation, action, env_state.actual_next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate_reward_fn),
            )
        )

        # Buffers
        self.disc_buffer = ParamsBuffer.create(self.discriminator_state, self.disc_buffer_capacity)
        self.etas_buffer = EtasBuffer.create(jnp.ones((self.nr_steps, self.nr_envs)), self.disc_buffer_capacity-1)
        self.policy_buffer = ParamsBuffer.create(self.policy_state, self.disc_buffer_capacity) # only used if on_demand_etas == True
        if self.on_demand_etas:
            self.policy_buffer = ParamsBuffer.add(self.policy_buffer, self.policy_state)

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

 
    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            # Expert demonstrations
            def _prepare_expert_data():
                return prepare_expert_data(self.data_path)
            
            demonstrations = jax.experimental.io_callback(_prepare_expert_data, expert_data_spec(num_samples=self.num_data_samples, state_dim=self.train_env.single_observation_space.shape[0], action_dim=self.train_env.single_action_space.shape[0]))
            expert_states = demonstrations["states"]
            expert_next_states = demonstrations["next_states"]
            expert_actions = demonstrations["actions"]
            expert_absorbing = demonstrations["absorbing"].flatten()

            # Set up carry objects
            policy_state = self.policy_state
            critic_state = self.critic_state
            discriminator_state = self.discriminator_state
            reward_fn_state = self.reward_fn_state
            disc_buffer = self.disc_buffer
            etas_buffer = self.etas_buffer
            policy_buffer = self.policy_buffer

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, critic_state, discriminator_state, reward_fn_state, \
                disc_buffer, etas_buffer, policy_buffer, \
                (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, discriminator_state, reward_fn_state, \
                    disc_buffer, etas_buffer, policy_buffer, \
                    (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                    env_state, key = learning_iteration_carry

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, env_state, key = single_rollout_carry
                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                        processed_action = self.get_processed_action(action)
                        value = self.critic.apply(critic_state.params, observation).squeeze(-1)

                        action_logstd = jnp.repeat(action_logstd[None, :], action_mean.shape[0], axis=0)

                        env_state = self.train_env.step(env_state, processed_action)
                        transition = (observation, env_state.actual_next_observation, action, env_state.reward, value, env_state.terminated, log_prob, action_mean, action_logstd, env_state.info)

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)
                            
                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, (policy_state, critic_state, env_state, key), None, self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, log_probs, old_action_means, old_action_logstd, infos = batch

                    """ TRIRL Components """
                    # Update discriminator
                    def trirl_loss_fn(discriminator_params, state, action, expert_state, expert_action, label, expert_label, next_state=None, absorbing=None, expert_next_state=None, expert_absorbing=None):
                        logits = self.discriminator.apply(discriminator_params, state, action, next_state, absorbing)
                        expert_logits = self.discriminator.apply(discriminator_params, expert_state, expert_action, expert_next_state, expert_absorbing)
                        bce_agent = optax.sigmoid_binary_cross_entropy(logits, label).mean()
                        bce_expert = optax.sigmoid_binary_cross_entropy(expert_logits, expert_label).mean()

                        # Gradient penalty on an interpolated sample between the expert and agent
                        alpha = self.gp_alpha
                        gp_lambda = self.gp_lambda
                        interpolated_state = alpha * expert_state + (1 - alpha) * state
                        interpolated_action = alpha * expert_action + (1 - alpha) * action
                        interpolated_next_state = alpha * expert_next_state + (1 - alpha) * next_state
                        interpolated_abs = 0.0 * expert_absorbing # assume interpolated state to be non-absorbing                
                        grad_state, grad_action, grad_next_state = jax.grad(lambda s, a, sn, ab: jnp.sum(self.discriminator.apply(discriminator_params, s, a, sn, ab)), argnums=(0, 1, 2))(interpolated_state, interpolated_action, interpolated_next_state, interpolated_abs)
                        grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_state)) + jnp.sum(jnp.square(grad_action)))
                        gp = (grad_norm - 1.0) ** 2

                        bce_loss = bce_agent + bce_expert + gp_lambda * gp
                        metrics = {
                            "loss/discriminator_loss": bce_loss,
                            "loss/discriminator_agent_loss": bce_agent,
                            "loss/discriminator_expert_loss": bce_expert,
                            "loss/discriminator_gp": gp
                        }
                        return bce_loss, (metrics)

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)

                    # Expert batch
                    key, shuffle_key = jax.random.split(key)
                    perm = jax.random.permutation(shuffle_key, expert_states.shape[0])
                    expert_states = expert_states[perm]
                    expert_actions = expert_actions[perm]
                    batch_expert_states = expert_states[:self.batch_size]
                    batch_expert_actions = expert_actions[:self.batch_size]
                    expert_labels = jnp.ones((self.batch_size, 1), dtype=jnp.float32)
                    rollout_labels = jnp.zeros((self.batch_size, 1), dtype=jnp.float32)

                    batch_next_states = next_states.reshape((-1,) + self.os_shape)
                    batch_absorbing = terminations.reshape(-1)
                    expert_next_states = expert_next_states[perm]
                    expert_absorbing = expert_absorbing[perm]
                    batch_expert_next_states = expert_next_states[:self.batch_size]
                    batch_expert_absorbing = expert_absorbing[:self.batch_size]

                    vmap_trirl_loss_fn = jax.vmap(trirl_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_trirl_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_trirl_loss_fn(*a, **k))
                    grad_trirl_loss_fn = jax.value_and_grad(mean_vmapped_trirl_loss_fn, argnums=(0), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices_disc = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs_disc, 1))
                    batch_indices_disc = jax.random.permutation(subkey, batch_indices_disc, axis=1, independent=True)
                    batch_indices_disc = batch_indices_disc.reshape((self.nr_epochs_disc * self.nr_minibatches, self.minibatch_size))

                    def trirl_minibatch_update(carry, minibatch_indices_disc):

                        discriminator_state = carry

                        # TRIRL UPDATE
                        (trirl_loss, (metrics)), (discriminator_gradients) = grad_trirl_loss_fn(
                            discriminator_state.params,
                            batch_states[minibatch_indices_disc],
                            batch_actions[minibatch_indices_disc],
                            batch_expert_states[minibatch_indices_disc],
                            batch_expert_actions[minibatch_indices_disc],
                            rollout_labels[minibatch_indices_disc],
                            expert_labels[minibatch_indices_disc],
                            batch_next_states[minibatch_indices_disc],
                            batch_absorbing[minibatch_indices_disc],
                            batch_expert_next_states[minibatch_indices_disc],
                            batch_expert_absorbing[minibatch_indices_disc],
                        )

                        discriminator_state = discriminator_state.apply_gradients(grads=discriminator_gradients)
                        carry = (discriminator_state)

                        return carry, (metrics)
                    
                    init_carry = (discriminator_state)
                    carry, (disc_optimization_metrics) = jax.lax.scan(trirl_minibatch_update, init_carry, batch_indices_disc)
                    discriminator_state = carry
                    
                    """ Reward Correction """
                    
                    # Define Apply Functions
                    def _get_log_density_ratio(inputs, discriminator_state):
                        state, action, next_state, absorbing = inputs
                        logits = self.discriminator.apply(discriminator_state.params, state, action, next_state, absorbing)
                        return logits
                    
                    def _plcy_apply_fn(inputs, policy_state):
                        state = inputs
                        action_mean, action_logstd = self.policy.apply(policy_state.params, state)
                        return action_mean, action_logstd

                    _get_log_density_ratio = jax.vmap(_get_log_density_ratio, in_axes=(0, None), out_axes=0)
                    _plcy_apply_fn = jax.vmap(_plcy_apply_fn, in_axes=(0, None), out_axes=(0, 0))
                    
                    chunked_correct = make_chunked_ensemble_rew_correct(
                        _get_log_density_ratio, 
                        nr_steps=self.nr_steps, 
                        nr_envs=self.nr_envs, 
                        epsilon=self.epsilon, 
                        beta=self.beta,
                        entropy_coef=self.entropy_coef,
                        maximum_eta=not self.maximum_eta
                    )

                    if self.on_demand_etas:
                        policy_params_sampled, level = ParamsBuffer.sample(policy_buffer) # get the whole buffer, only use till level during projection
                        chunked_compute_etas = make_chunked_compute_etas(
                            policy_apply_fn=_plcy_apply_fn,
                            kl_cov_proj=kl_cov_proj,
                            nr_steps=self.nr_steps, 
                            nr_envs=self.nr_envs,
                            mean_bound=self.mean_bound,
                            cov_bound=self.cov_bound,
                        )

                        proj_etas = chunked_compute_etas(
                            policy_params_sampled,
                            states.reshape((-1,) + self.os_shape),
                            level,
                            chunk_size=self.chunk_size,
                            buffer_size=self.disc_buffer_capacity
                        )
                    else:
                        proj_etas, _ = EtasBuffer.sample(etas_buffer) # etas are zero padded during projection

                    # Add new discriminator params to the buffer
                    disc_buffer = ParamsBuffer.add(disc_buffer, discriminator_state)
                    disc_params_sampled, level = ParamsBuffer.sample(disc_buffer) # get the whole buffer, only use till level during correction
                    init_corr_reward = jnp.zeros_like(rewards)

                    corr_reward = chunked_correct(
                        disc_params_sampled,
                        (states.reshape((-1,) + self.os_shape),
                        actions.reshape((-1,) + self.as_shape),
                        next_states.reshape((-1,) + self.os_shape),
                        terminations.flatten(),
                        ),
                        proj_etas,
                        init_corr_reward,
                        level,
                        chunk_size=self.chunk_size,
                    ).reshape(rewards.shape)

                    if self.handle_absorbing_states:
                        corr_reward_absorbing_state = chunked_correct(
                            disc_params_sampled,
                            (next_states.reshape((-1,) + self.os_shape),
                            actions.reshape((-1,) + self.as_shape),
                            next_states.reshape((-1,) + self.os_shape),
                            jnp.ones_like(terminations.flatten()),
                            ),
                            proj_etas,
                            jnp.zeros_like(rewards),
                            level,
                            chunk_size=self.chunk_size,
                        ).reshape(rewards.shape)
                    else:
                        corr_reward_absorbing_state = jnp.asarray(0.0)


                    def reward_approximator_loss_fn(reward_fn_params, state, action, target_reward, next_state=None):
                        logits = self.reward_fn.apply(reward_fn_params, state, action, next_state)
                        mse = optax.squared_error(logits, target_reward).mean()

                        metrics = {
                            "loss/reward_fn_loss": mse,
                        }
                        return mse, (metrics)

                    if self.reward_fn_approximator:
                        batch_states = states.reshape((-1,) + self.os_shape)
                        batch_actions = actions.reshape((-1,) + self.as_shape)
                        batch_target_reward = corr_reward.reshape((-1,) + (1,))
                        batch_next_states = next_states.reshape((-1,) + self.os_shape)

                        vmap_loss_fn = jax.vmap(reward_approximator_loss_fn, in_axes=(None, 0, 0, 0, 0), out_axes=0)
                        safe_mean = lambda x: jnp.mean(x) if x is not None else x
                        mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                        grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0), has_aux=True)

                        key, subkey = jax.random.split(key)
                        batch_indices_rew = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs_rew, 1))
                        batch_indices_rew = jax.random.permutation(subkey, batch_indices_rew, axis=1, independent=True)
                        batch_indices_rew = batch_indices_rew.reshape((self.nr_epochs_rew * self.nr_minibatches, self.minibatch_size))

                        def reward_approximator_minibatch_update(carry, minibatch_indices_rew):

                            reward_fn_state = carry

                            # REWARD UPDATE
                            (rew_fn_loss, (metrics)), (rew_fn_gradients) = grad_loss_fn(
                                reward_fn_state.params,
                                batch_states[minibatch_indices_rew],
                                batch_actions[minibatch_indices_rew],
                                batch_target_reward[minibatch_indices_rew],
                                batch_next_states[minibatch_indices_rew],
                            )

                            reward_fn_state = reward_fn_state.apply_gradients(grads=rew_fn_gradients)
                            carry = (reward_fn_state)

                            return carry, (metrics)
                        
                        init_carry = (reward_fn_state)
                        carry, (reward_approximator_optimization_metrics) = jax.lax.scan(reward_approximator_minibatch_update, init_carry, batch_indices_rew)
                        reward_fn_state = carry

                        def get_reward_prediction(reward_fn_params, state, action, next_state):
                            logits = self.reward_fn.apply(reward_fn_params, state, action, next_state)
                            return logits
                        
                        get_reward_prediction = jax.vmap(get_reward_prediction, in_axes=(None, 0, 0, 0), out_axes=0)

                        corr_reward = get_reward_prediction(reward_fn_state.params, states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape), next_states.reshape((-1,) + self.os_shape))
                        corr_reward = corr_reward.reshape(rewards.shape)
                        corr_reward_absorbing_state = get_reward_prediction(reward_fn_state.params, next_states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape), next_states.reshape((-1,) + self.os_shape))
                        corr_reward_absorbing_state = corr_reward_absorbing_state.reshape(rewards.shape)
                    else:
                        reward_approximator_optimization_metrics = {}
                        
                    
                    """ PPO + DTRL """
                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
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

                    def calculate_gae_advantages_absorbing(critic_state, next_states, rewards, rewards_next_state, values, terminations):
                        """
                        Correctly handle absorbing state value and entropy (instead of setting to 0.0)
                        """
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        terminal_tail = (self.gamma / (1.0 - self.gamma)) * (rewards_next_state + self.entropy_coef * self.H_terminal)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) + (terminations * terminal_tail) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                
                    if self.handle_absorbing_states:
                        advantages, returns = calculate_gae_advantages_absorbing(critic_state, next_states, corr_reward, corr_reward_absorbing_state, values, terminations)
                    else:
                        advantages, returns = calculate_gae_advantages(critic_state, next_states, corr_reward, values, terminations)


                    # Optimizing
                    def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b, old_mean_b, old_logstd_b, global_step):
                        # Policy eval
                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)

                        # Entropy projection (project std to satisfy an entropy constraint)
                        old_logstd_b = old_logstd_b[-1] # (as_shape, )
                        old_logstd_b = jnp.expand_dims(old_logstd_b, 0) # shape (1, as_shape)
                        old_entropy = jnp.sum(old_logstd_b + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
                        tau = 0.5
                        beta = old_entropy * tau ** (10 * jnp.clip(global_step, max=self.total_timesteps) / self.total_timesteps)
                        # Entropy projection before KL projection
                        # action_logstd = entropy_projection(action_logstd, beta, dim=self.dim)

                        # Projection layer
                        action_std = jnp.exp(action_logstd)
                        old_action_mean = old_mean_b
                        old_action_std = jnp.exp(old_logstd_b)

                        proj_action_mean, proj_action_std, eta_mu, eta_cov, kl_mean_part, post_proj_kl_mean_part, kl_cov_part, post_proj_kl_cov_part = kl_projection(action_mean, action_std, old_action_mean, old_action_std, self.mean_bound, self.cov_bound)
                        # Entropy projection after KL projection
                        proj_action_logstd = jnp.log(proj_action_std)
                        proj_action_logstd = entropy_projection(proj_action_logstd, beta, dim=self.dim)

                        # Trust Region Loss (amortized optimization)
                        proj_action_mean_det = jax.lax.stop_gradient(proj_action_mean) # detaching to prevent gradient flow through projection from the regression term
                        proj_action_std_det  = jax.lax.stop_gradient(proj_action_std)
                        tr_loss_maha = 0.5 * jnp.sum(((proj_action_mean_det - action_mean)/ proj_action_std_det) ** 2, axis=1)
                        tr_loss_cov_part = 0.5 * jnp.sum(2.0 * (jnp.log(proj_action_std_det) - jnp.log(action_std)) + (action_std/proj_action_std_det)**2 - 1.0, axis=1)
                        trust_region_loss = tr_loss_maha + tr_loss_cov_part

                        action_mean = proj_action_mean
                        action_logstd = proj_action_logstd
                        action_std = jnp.exp(action_logstd)

                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum(1)
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                        entropy = self.entropy_coef * entropy # scaling to reduce critic loss (see reward correction for more)
                        
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
                        loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss + self.trust_region_coef * trust_region_loss

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "loss/trust_region_loss": trust_region_loss,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                            "projection/eta_mu": eta_mu,
                            "projection/eta_cov": eta_cov,
                            "projection/unprojected_kl_mean": kl_mean_part,
                            "projection/projected_kl_mean": post_proj_kl_mean_part,
                            "projection/unprojected_kl_cov": kl_cov_part,
                            "projection/projected_kl_cov": post_proj_kl_cov_part,
                        }

                        return loss, (metrics, jnp.maximum(eta_mu, eta_cov))
                    
                    # Gradients must only flow through the predicted ones
                    old_action_means = jax.lax.stop_gradient(old_action_means)
                    old_action_logstd = jax.lax.stop_gradient(old_action_logstd)
                    old_action_logstd = old_action_logstd.squeeze()

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)
                    batch_action_means = old_action_means.reshape((-1,) + self.as_shape)
                    batch_action_logstds = old_action_logstd.reshape((-1,) + self.as_shape)

                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0, 0, None, None), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: (lambda out: (tree.map_structure(safe_mean, out[0]), (tree.map_structure(safe_mean, out[1][0]), out[1][1])))(vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def dtrl_minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                        combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                        global_step = combined_learning_iteration_step * self.nr_steps * self.nr_envs

                        (loss, (metrics, etas)), (policy_gradients, critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages,
                            batch_action_means[minibatch_indices],
                            batch_action_logstds[minibatch_indices], 
                            global_step
                        )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                        carry = (policy_state, critic_state)
                        return carry, (metrics, etas)

                    init_carry = (policy_state, critic_state)
                    carry, (dtrl_optimization_metrics, etas) = jax.lax.scan(dtrl_minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state = carry

                    dtrl_optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    dtrl_optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    dtrl_optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))

                    # Calculate max per-epoch eta
                    etas = etas.reshape((self.nr_epochs, self.nr_steps, self.nr_envs))
                    etas = etas.max(axis=0)
                    etas_buffer = EtasBuffer.add(etas_buffer, etas)
                    dtrl_optimization_metrics["projection/max_eta"] = jnp.max(etas)

                    # Add updated policy to buffer
                    if self.on_demand_etas:
                        policy_buffer = ParamsBuffer.add(policy_buffer, policy_state)

                    # Logging
                    combined_metrics = {**infos, **disc_optimization_metrics, **reward_approximator_optimization_metrics, **dtrl_optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = combined_learning_iteration_step.item() * self.nr_epochs * self.nr_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id))
                    
                    return (policy_state, critic_state, discriminator_state, reward_fn_state, disc_buffer, etas_buffer, policy_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, discriminator_state, reward_fn_state,
                                                                                disc_buffer, etas_buffer, policy_buffer,
                                                                                (expert_states, expert_actions, expert_next_states, expert_absorbing),
                                                                                env_state, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))

                policy_state, critic_state, discriminator_state, reward_fn_state, \
                disc_buffer, etas_buffer, policy_buffer, \
                (expert_states, expert_actions, expert_next_states, expert_absorbing), \
                env_state, key = learning_iteration_carry


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

                return (policy_state, critic_state, discriminator_state, reward_fn_state, disc_buffer, etas_buffer, policy_buffer, (expert_states, expert_actions, expert_next_states, expert_absorbing), env_state, key), None


            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, discriminator_state, reward_fn_state,
                                                                  disc_buffer, etas_buffer, policy_buffer,
                                                                  (expert_states, expert_actions, expert_next_states, expert_absorbing),
                                                                  env_state, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))
            

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")
    

    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_wandb:
            self.wandb_log_cache[name] = value
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
        model = TRIRL_DTRL(config, train_env, eval_env, run_path, writer)

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
