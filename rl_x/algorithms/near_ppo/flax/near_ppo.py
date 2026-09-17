import os
import shutil
import tempfile
import json
from functools import partial
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

from rl_x.algorithms.near_ppo.flax.general_properties import GeneralProperties
from rl_x.algorithms.near_ppo.flax.policy import get_policy
from rl_x.algorithms.near_ppo.flax.critic import get_critic
from rl_x.algorithms.near_ppo.flax.energy_function import get_energyfn
from rl_x.algorithms.near_ppo.flax.data_utils import prepare_expert_data

rlx_logger = logging.getLogger("rl_x")


class NEAR_PPO:
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
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = getattr(self.train_env, "horizon", 1000)

        self.data_path = config.algorithm.data_path
        self.batch_size_ncsn = config.algorithm.batch_size_ncsn
        self.minibatch_size_ncsn = config.algorithm.minibatch_size_ncsn
        self.total_samples_ncsn = config.algorithm.total_samples_ncsn
        self.nr_epochs_ncsn = config.algorithm.nr_epochs_ncsn
        self.anneal_power_ncsn = config.algorithm.anneal_power_ncsn
        self.sigma_begin_ncsn = config.algorithm.sigma_begin_ncsn
        self.sigma_end_ncsn = config.algorithm.sigma_end_ncsn
        self.L_ncsn = config.algorithm.L_ncsn
        self.nr_hidden_units_encoder_ncsn = config.algorithm.nr_hidden_units_encoder_ncsn
        self.nr_hidden_units_decoder_ncsn = config.algorithm.nr_hidden_units_decoder_ncsn
        self.learning_rate_ncsn = config.algorithm.learning_rate_ncsn
        self.sigma_inference_ncsn = config.algorithm.sigma_inference_ncsn
        self.ncsnv1 = config.algorithm.ncsnv1
        self.env_reward_frac = config.algorithm.env_reward_frac
        self.handle_absorbing_states = config.algorithm.handle_absorbing_states
        self.nr_minibatches_ncsn = self.batch_size_ncsn // self.minibatch_size_ncsn
        self.state_based = config.algorithm.state_based

        if config.runner.mode == "train" and self.minibatch_size > self.batch_size:
            raise ValueError("Minibatch size must not be larger than batch size")

        if self.minibatch_size_ncsn > self.batch_size_ncsn:
            raise ValueError("NCSN minibatch size must not be larger than NCSN batch size")

        if config.runner.mode == "train" and self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, energyfn_key, reset_key = jax.random.split(self.key, 5)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)
        self.energyfn = get_energyfn(config, self.train_env, ncsnv1=self.ncsnv1)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate and config.runner.mode == "train" else self.learning_rate
        learning_rate_ncsn = self.learning_rate_ncsn

        self.key, sampling_key = jax.random.split(self.key)
        initial_observation = jnp.asarray(self.train_env.reset()[0][:1], dtype=jnp.float32)
        self.H_terminal = jnp.sum(jnp.log(self.train_env.single_action_space.high - self.train_env.single_action_space.low)) # terminal entropy assuming uniform policy

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, initial_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        if config.runner.mode == "test":
            return

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, initial_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        state = initial_observation
        next_state = initial_observation
        action = jax.random.uniform(sampling_key, (1,) + self.as_shape, minval=jnp.asarray(self.train_env.single_action_space.low), maxval=jnp.asarray(self.train_env.single_action_space.high))
        cond = 1.0
        if self.state_based:
            init_input = jnp.concatenate([state.flatten(), next_state.flatten()])
        else:
            init_input = jnp.concatenate([state.flatten(), action.flatten()])

        self.energyfn_state = TrainState.create(
            apply_fn=self.energyfn.apply,
            params=self.energyfn.init(energyfn_key, init_input, cond),
            tx=optax.chain(
                optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate_ncsn),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.expert_data = jax.tree.map(jnp.asarray, prepare_expert_data(self.data_path))


    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(self, policy_state, critic_state, observation, key):
        key, subkey = jax.random.split(key)
        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
        action_std = jnp.exp(action_logstd)
        action = action_mean + action_std * jax.random.normal(subkey, action_mean.shape)
        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
        value = self.critic.apply(critic_state.params, observation).squeeze(-1)
        return self.get_processed_action(action), action, value, log_prob, action_mean, jnp.repeat(action_logstd[None, :], action_mean.shape[0], axis=0), key


    @partial(jax.jit, static_argnums=0)
    def pretrain_update(self, energyfn_state, expert_data, key):
        expert_states = expert_data["states"]
        expert_actions = expert_data["actions"]
        expert_next_states = expert_data["next_states"]
        def ncsn_loss_fn(energyfn_params, expert_state, expert_action, expert_next_state, key):
            """
            Denoising Score Matching
            """

            key, label_key = jax.random.split(key)
            sigmas = jnp.exp(jnp.linspace(jnp.log(self.sigma_begin_ncsn), jnp.log(self.sigma_end_ncsn), self.L_ncsn))
            conds = jnp.arange(self.L_ncsn)
            used_cond = jax.random.choice(label_key, conds)
            used_sigma = sigmas[used_cond]

            if self.state_based:
                sample = jnp.concatenate([expert_state.flatten(), expert_next_state.flatten()])
            else:
                sample = jnp.concatenate([expert_state.flatten(), expert_action.flatten()])
            perturbed_sample = sample + jax.random.normal(key, shape=sample.shape) * used_sigma
            target = - 1 / (used_sigma ** 2) * (perturbed_sample - sample)

            if self.ncsnv1:
                pred_score = jax.grad(lambda x, cond: jnp.sum(self.energyfn.apply(energyfn_params, x, cond)), argnums=(0))(perturbed_sample, used_cond)
            else:
                pred_score = jax.grad(lambda x, cond: jnp.sum(self.energyfn.apply(energyfn_params, x, cond)), argnums=(0))(perturbed_sample, used_sigma)
            dsm_loss = jnp.mean((1/2.) * ((pred_score - target) ** 2).sum() * (used_sigma ** self.anneal_power_ncsn))

            metrics = {
                "loss/energyfn_loss": dsm_loss,
            }
            return dsm_loss, (metrics)

        key, expert_key = jax.random.split(key)
        expert_indices = jax.random.randint(expert_key, (self.batch_size_ncsn,), 0, expert_states.shape[0])
        batch_expert_states = expert_states[expert_indices]
        batch_expert_actions = expert_actions[expert_indices]
        batch_expert_next_states = expert_next_states[expert_indices]

        vmap_ncsn_loss_fn = jax.vmap(ncsn_loss_fn, in_axes=(None, 0, 0, 0, 0), out_axes=0)
        safe_mean = lambda x: jnp.mean(x) if x is not None else x
        mean_vmapped_ncsn_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_ncsn_loss_fn(*a, **k))
        grad_ncsn_loss_fn = jax.value_and_grad(mean_vmapped_ncsn_loss_fn, argnums=(0), has_aux=True)

        key, subkey = jax.random.split(key)
        batch_indices_ncsn = jnp.tile(jnp.arange(self.batch_size_ncsn), (self.nr_epochs_ncsn, 1))
        batch_indices_ncsn = jax.random.permutation(subkey, batch_indices_ncsn, axis=1, independent=True)
        batch_indices_ncsn = batch_indices_ncsn[:, :self.nr_minibatches_ncsn * self.minibatch_size_ncsn]
        batch_indices_ncsn = batch_indices_ncsn.reshape((self.nr_epochs_ncsn * self.nr_minibatches_ncsn, self.minibatch_size_ncsn))

        def ncsn_minibatch_update(carry, minibatch_indices_ncsn):
            energyfn_state, key = carry

            key, label_key = jax.random.split(key)
            mb_keys = jax.random.split(label_key, self.minibatch_size_ncsn)

            (near_loss, (metrics)), (energyfn_gradients) = grad_ncsn_loss_fn(
                energyfn_state.params,
                batch_expert_states[minibatch_indices_ncsn],
                batch_expert_actions[minibatch_indices_ncsn],
                batch_expert_next_states[minibatch_indices_ncsn],
                mb_keys,
            )

            energyfn_state = energyfn_state.apply_gradients(grads=energyfn_gradients)
            metrics["gradients/energyfn_grad_norm"] = optax.global_norm(energyfn_gradients)

            carry = (energyfn_state, key)

            return carry, (metrics)

        init_carry = (energyfn_state, key)
        carry, (ncsn_optimization_metrics) = jax.lax.scan(ncsn_minibatch_update, init_carry, batch_indices_ncsn)
        energyfn_state, key = carry


        return energyfn_state, tree.map_structure(jnp.mean, ncsn_optimization_metrics), key


    @partial(jax.jit, static_argnums=0)
    def update(self, train_state, batch, expert_data, key, learning_iteration_step):
        policy_state, critic_state, energyfn_state = train_state
        states, next_states, actions, rewards, values, terminations, log_probs, old_action_means, old_action_logstd, infos = batch
        expert_states = expert_data["states"]
        expert_actions = expert_data["actions"]
        expert_next_states = expert_data["next_states"]
        expert_absorbing = expert_data["absorbing"]


        """ Reward Computation """

        def get_energy_reward(inputs, cond, energyfn_state, last_update_mean_energy):
            state, action, next_state, absorbing = inputs
            if self.state_based:
                sample = jnp.concatenate((state, next_state))
            else:
                sample = jnp.concatenate((state, action))
            energy = jax.lax.stop_gradient(self.energyfn.apply(energyfn_state.params, sample, cond))
            energy_reward = 10 * jnp.tanh((energy - last_update_mean_energy)/10)

            return energy_reward, energy

        get_energy_reward = jax.vmap(get_energy_reward, in_axes=(0, None, None, None), out_axes=0)


        sigmas = jnp.exp(jnp.linspace(jnp.log(self.sigma_begin_ncsn), jnp.log(self.sigma_end_ncsn), self.L_ncsn))

        if self.state_based:
            samples = jnp.concatenate([states.reshape((-1,) + self.os_shape), next_states.reshape((-1,) + self.os_shape)], axis=1)
        else:
            samples = jnp.concatenate([states.reshape((-1,) + self.os_shape), actions.reshape((-1,) + self.as_shape)], axis=1)

        if self.ncsnv1:
            energy_reward = jnp.squeeze(jax.lax.stop_gradient(self.energyfn.apply(energyfn_state.params, samples, self.sigma_inference_ncsn))).reshape(rewards.shape)
        else:
            energy_reward = jnp.squeeze(jax.lax.stop_gradient(self.energyfn.apply(energyfn_state.params, samples, sigmas[self.sigma_inference_ncsn]))).reshape(rewards.shape)


        energy_reward_absorbing_state = jnp.asarray(0.0)

        energy_reward = self.env_reward_frac * rewards + (1 - self.env_reward_frac) * energy_reward

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
            advantages, returns = calculate_gae_advantages_absorbing(critic_state, next_states, energy_reward, energy_reward_absorbing_state, values, terminations)
        else:
            advantages, returns = calculate_gae_advantages(critic_state, next_states, energy_reward, values, terminations)

        def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):

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

            new_value = self.critic.apply(critic_params, state_b)
            critic_loss = 0.5 * (new_value - return_b) ** 2

            loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

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
        batch_action_means = old_action_means.reshape((-1,) + self.as_shape)

        vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
        safe_mean = lambda x: jnp.mean(x) if x is not None else x
        mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
        grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

        key, subkey = jax.random.split(key)
        batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
        batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
        batch_indices = batch_indices[:, :self.nr_minibatches * self.minibatch_size]
        batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

        def ppo_minibatch_update(carry, minibatch_indices):
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
                minibatch_advantages,
            )

            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            critic_state = critic_state.apply_gradients(grads=critic_gradients)

            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

            carry = (policy_state, critic_state)
            return carry, (metrics)

        init_carry = (policy_state, critic_state)
        carry, (ppo_optimization_metrics) = jax.lax.scan(ppo_minibatch_update, init_carry, batch_indices)
        policy_state, critic_state = carry

        ppo_optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
        ppo_optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
        ppo_optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))
        ppo_optimization_metrics["reward/sigma_level"] = np.array([self.sigma_inference_ncsn])
        ppo_optimization_metrics["reward/mean_energy"] = jnp.mean(energy_reward)

        combined_metrics = {**infos, **ppo_optimization_metrics}
        combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)


        return (policy_state, critic_state, energyfn_state), combined_metrics, key


    def train(self):
        for step in range(int(self.total_samples_ncsn // self.nr_epochs_ncsn // self.batch_size_ncsn)):
            self.energyfn_state, metrics, self.key = self.pretrain_update(self.energyfn_state, self.expert_data, self.key)
            self.start_logging((step + 1) * self.nr_epochs_ncsn * self.batch_size_ncsn)
            for metric_name, value in metrics.items():
                self.log(metric_name, float(value), (step + 1) * self.nr_epochs_ncsn * self.batch_size_ncsn)
            self.end_logging()

        observation, _ = self.train_env.reset()
        for learning_iteration_step in range(int(self.nr_updates)):
            started = time.time()
            transitions = []
            info_values = {}
            for step in range(self.nr_steps):
                processed_action, action, value, log_prob, mean, logstd, self.key = self.get_action_and_value(self.policy_state, self.critic_state, jnp.asarray(observation), self.key)
                next_observation, reward, terminated, truncated, info = self.train_env.step(np.asarray(processed_action))
                actual_next_observation = next_observation.copy()
                for index in np.flatnonzero(terminated | truncated):
                    actual_next_observation[index] = self.train_env.get_final_observation_at_index(info, index)
                for metric_name, values in self.train_env.get_logging_info_dict(info).items():
                    group = "rollout" if metric_name in ["episode_return", "episode_length"] else "env_info"
                    info_values.setdefault(f"{group}/{metric_name}", []).extend(np.asarray(values).reshape(-1))
                transitions.append((observation, actual_next_observation, action, reward, value, terminated, log_prob, mean, logstd))
                observation = next_observation
            batch = tuple(jnp.asarray(np.stack(values), dtype=jnp.float32) for values in zip(*transitions)) + ({},)
            train_state, metrics, self.key = self.update((self.policy_state, self.critic_state, self.energyfn_state), batch, self.expert_data, self.key, learning_iteration_step)
            self.policy_state, self.critic_state, self.energyfn_state = train_state
            global_step = (learning_iteration_step + 1) * self.batch_size
            metrics = {key: float(value) for key, value in metrics.items()}
            metrics.update({key: float(np.mean(values)) for key, values in info_values.items() if len(values)})
            metrics["time/sps"] = self.batch_size / (time.time() - started)
            metrics["steps/nr_env_steps"] = global_step
            if global_step % self.evaluation_and_save_frequency == 0:
                if self.evaluation_active:
                    metrics.update(self.evaluate())
                    if self.eval_env is self.train_env:
                        observation, _ = self.train_env.reset()
                if self.save_model:
                    self.save(self.policy_state, self.critic_state, self.energyfn_state)
            self.start_logging(global_step)
            for metric_name, value in metrics.items():
                self.log(metric_name, value, global_step)
            self.end_logging()


    def evaluate(self):
        observation, _ = self.eval_env.reset()
        returns = []
        lengths = []
        for step in range(self.horizon):
            mean, _ = self.policy.apply(self.policy_state.params, jnp.asarray(observation))
            observation, reward, terminated, truncated, info = self.eval_env.step(np.asarray(self.get_processed_action(mean)))
            for index in np.flatnonzero(terminated | truncated):
                returns.append(self.eval_env.get_final_info_value_at_index(info, "episode_return", index))
                lengths.append(self.eval_env.get_final_info_value_at_index(info, "episode_length", index))
        return {"eval/episode_return": float(np.mean(returns)) if returns else 0.0, "eval/episode_length": float(np.mean(lengths)) if lengths else 0.0}


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


    def save(self, policy_state, critic_state, energyfn_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "energyfn": energyfn_state,
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
        checkpoint_dir = tempfile.mkdtemp(prefix="rlx-checkpoint-")
        shutil.unpack_archive(config.runner.load_model, checkpoint_dir, "zip")

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if key != "name" and f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = NEAR_PPO(config, train_env, eval_env, run_path, writer)
        if config.runner.mode == "test":
            target = {"policy": model.policy_state}
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            checkpoint = checkpointer.restore(checkpoint_dir, args=orbax.checkpoint.args.PyTreeRestore(item=target, restore_args=orbax_utils.restore_args_from_target(target), partial_restore=True))
            model.policy_state = checkpoint["policy"]
            shutil.rmtree(checkpoint_dir)
            return model


        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "energyfn": model.energyfn_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.energyfn_state = checkpoint["energyfn"]

        shutil.rmtree(checkpoint_dir)

        return model


    def test(self, episodes):
        observation, _ = self.eval_env.reset()
        completed = 0
        while completed < episodes:
            mean, _ = self.policy.apply(self.policy_state.params, jnp.asarray(observation))
            observation, reward, terminated, truncated, info = self.eval_env.step(np.asarray(self.get_processed_action(mean)))
            for index in np.flatnonzero(terminated | truncated):
                completed += 1
                rlx_logger.info(f"Episode {completed} - Return: {self.eval_env.get_final_info_value_at_index(info, 'episode_return', index)}")
                if completed >= episodes:
                    break


    def general_properties():
        return GeneralProperties
