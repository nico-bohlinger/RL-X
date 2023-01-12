import os
import logging
import pickle
import random
import time
from collections import deque
import tree
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
import optax
import wandb
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tensorflow_probability.substrates.jax.distributions

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
        self.nr_samples = config.algorithm.nr_samples
        self.stability_epsilon = config.algorithm.stability_epsilon
        self.init_log_temperature = config.algorithm.init_log_temperature
        self.init_log_alpha_mean = config.algorithm.init_log_alpha_mean
        self.init_log_alpha_stddev = config.algorithm.init_log_alpha_stddev
        self.min_log_temperature = config.algorithm.min_log_temperature
        self.min_log_alpha = config.algorithm.min_log_alpha
        self.kl_epsilon = config.algorithm.kl_epsilon
        self.kl_epsilon_penalty = config.algorithm.kl_epsilon_penalty
        self.kl_epsilon_mean = config.algorithm.kl_epsilon_mean
        self.kl_epsilon_stddev = config.algorithm.kl_epsilon_stddev
        self.retrace_lambda = config.algorithm.retrace_lambda
        self.trace_length = config.algorithm.trace_length
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.logging_all_metrics = config.algorithm.logging_all_metrics
        self.logging_freq = config.algorithm.logging_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units

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
            log_alpha_stddev=jnp.full(dual_variable_shape, self.init_log_alpha_stddev, dtype=jnp.float32),
            log_penalty_temperature=jnp.full([1], self.init_log_temperature, dtype=jnp.float32)
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
        def get_action_and_log_prob(policy_params: flax.core.FrozenDict, state: np.ndarray, key: jax.random.PRNGKey):
            dist = self.policy.apply(policy_params, state)
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            log_prob = dist.log_prob(action)
            return action, log_prob, key


        @jax.jit
        def get_log_prob(policy_params: flax.core.FrozenDict, state: np.ndarray, action: np.ndarray):
            dist = self.policy.apply(policy_params, state)
            return dist.log_prob(action)


        @jax.jit
        def update(train_state: TrainingState, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, log_probs: np.ndarray, key: jax.random.PRNGKey):
            def loss_fn(agent_params: flax.core.FrozenDict, dual_params: flax.core.FrozenDict, agent_target_params: flax.core.FrozenDict,
                        states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, log_probs: np.ndarray, k1: jax.random.PRNGKey, k2: jax.random.PRNGKey
                ):
                # Compute predictions
                pred_policy = self.policy.apply(agent_params.policy_params, states)
                q_value_pred = self.vector_critic.apply(agent_params.critic_params, states[:-1], actions[:-1]).squeeze((0, 2))


                # Compute targets
                target_policy = self.policy.apply(agent_target_params.policy_params, states)

                a_improvement = target_policy.sample(self.nr_samples, seed=k1)

                vmap_critic_call = jax.vmap(self.vector_critic.apply, in_axes=(None, None, 0))
                q_improvement = vmap_critic_call(agent_target_params.critic_params, states, a_improvement).squeeze((1, 3))

                eval_policy = target_policy

                a_evaluation = eval_policy.sample(self.nr_samples, seed=k2)

                value_target = vmap_critic_call(agent_target_params.critic_params, states, a_evaluation).squeeze((1, 3))
                value_target = jnp.mean(value_target, axis=0)

                ###
                # Retrace calculation defined in rlax and used by acme: https://github.com/deepmind/rlax/blob/master/rlax/_src/multistep.py#L380#L433
                q_t = self.vector_critic.apply(agent_target_params.critic_params, states, actions).squeeze((0, 2))[1:-1]
                v_t = value_target[1:]
                r_t = rewards[:-1]
                discount_t = self.gamma * (1 - dones[:-1])
                log_rhos = target_policy.log_prob(action) - log_probs
                c_t = self.retrace_lambda * jnp.minimum(1.0, jnp.exp(log_rhos[1:-1]))

                g = r_t[-1] + discount_t[-1] * v_t[-1]

                def _body(acc, xs):
                    reward, discount, c, v, q = xs
                    acc = reward + discount * (v - c * q + c * acc)
                    return acc, acc
                
                _, returns = jax.lax.scan(_body, g, (r_t[:-1], discount_t[:-1], c_t, v_t[:-1], q_t), reverse=True)
                returns = jnp.concatenate([returns, g[jnp.newaxis]], axis=0)
                ###

                q_value_target = jax.lax.stop_gradient(returns)


                # Compute policy loss

                # Cast `MultivariateNormalDiag`s to Independent Normals. Allows to satisfy KL constraints per-dimension.
                target_policy = tfd.Independent(tfd.Normal(target_policy.mean(), target_policy.stddev()))
                current_policy = tfd.Independent(tfd.Normal(pred_policy.mean(), pred_policy.stddev()))

                # Convert from log. Use softplus for stability
                temperature = jax.nn.softplus(dual_params.log_temperature) + self.stability_epsilon
                alpha_mean = jax.nn.softplus(dual_params.log_alpha_mean) + self.stability_epsilon
                alpha_stddev = jax.nn.softplus(dual_params.log_alpha_stddev) + self.stability_epsilon
                penalty_temperature = jax.nn.softplus(dual_params.log_penalty_temperature) + self.stability_epsilon

                current_mean = current_policy.distribution.mean()
                current_stddev = current_policy.distribution.stddev()
                target_mean = target_policy.distribution.mean()
                target_stddev = target_policy.distribution.stddev()

                # Optimize the dual function. Equation (9) from the paper
                tempered_q_values = jax.lax.stop_gradient(q_improvement) / temperature
                normalized_weights = jax.lax.stop_gradient(jax.nn.softmax(tempered_q_values, axis=0))
                q_logsumexp = jax.scipy.special.logsumexp(tempered_q_values, axis=0)
                loss_temperature = temperature * (self.kl_epsilon + jnp.mean(q_logsumexp) - jnp.log(self.nr_samples))
                # Estimate actualized KL
                kl_nonparametric = jnp.sum(normalized_weights * jnp.log(self.nr_samples * normalized_weights + self.stability_epsilon), axis=0)

                # Penalty for violating the action constraint
                diff_out_of_bound = a_improvement - jnp.clip(a_improvement, -1.0, 1.0)
                cost_out_of_bound = -jnp.linalg.norm(diff_out_of_bound, axis=-1)
                tempered_cost = jax.lax.stop_gradient(cost_out_of_bound) / penalty_temperature
                penalty_normalized_weights = jax.lax.stop_gradient(jax.nn.softmax(tempered_cost, axis=0))
                penalty_logsumexp = jax.scipy.special.logsumexp(tempered_cost, axis=0)
                loss_penalty_temperature = penalty_temperature * (self.kl_epsilon_penalty + jnp.mean(penalty_logsumexp) - jnp.log(self.nr_samples))
                # Estimate actualized KL
                penalty_kl_nonparametric = jnp.sum(penalty_normalized_weights * jnp.log(self.nr_samples * penalty_normalized_weights + self.stability_epsilon), axis=0)

                normalized_weights += penalty_normalized_weights
                loss_temperature += loss_penalty_temperature

                # Calculate KL loss for policy

                # Decompose current policy into fixed-mean & fixed-stddev distributions
                fixed_stddev_dist = tfd.Independent(tfd.Normal(loc=current_mean, scale=target_stddev))
                fixed_mean_dist = tfd.Independent(tfd.Normal(loc=target_mean, scale=current_stddev))

                loss_policy_mean = jnp.mean(-jnp.sum(fixed_stddev_dist.log_prob(a_improvement) * normalized_weights, axis=0))
                loss_policy_stddev = jnp.mean(-jnp.sum(fixed_mean_dist.log_prob(a_improvement) * normalized_weights, axis=0))

                kl_mean = target_policy.distribution.kl_divergence(fixed_stddev_dist.distribution)
                kl_stddev = target_policy.distribution.kl_divergence(fixed_mean_dist.distribution)

                mean_kl_mean = jnp.mean(kl_mean, axis=0)
                loss_kl_mean = jnp.sum(jax.lax.stop_gradient(alpha_mean) * mean_kl_mean)
                loss_alpha_mean = jnp.sum(alpha_mean * (self.kl_epsilon_mean - jax.lax.stop_gradient(mean_kl_mean)))

                mean_kl_stddev = jnp.mean(kl_stddev, axis=0)
                loss_kl_stddev = jnp.sum(jax.lax.stop_gradient(alpha_stddev) * mean_kl_stddev)
                loss_alpha_stddev = jnp.sum(alpha_stddev * (self.kl_epsilon_stddev - jax.lax.stop_gradient(mean_kl_stddev)))

                # Combine losses
                unconst_policy_loss = loss_policy_mean + loss_policy_stddev
                kl_penalty_loss = loss_kl_mean + loss_kl_stddev
                alpha_loss = loss_alpha_mean + loss_alpha_stddev
                policy_loss = unconst_policy_loss + kl_penalty_loss + alpha_loss + loss_temperature


                # Compute critic loss
                critic_loss = jnp.mean(0.5 * jnp.square(q_value_target - q_value_pred))

                loss = policy_loss + critic_loss


                # Create metrics
                metrics = {
                    "dual/alpha_mean": jnp.mean(alpha_mean),
                    "dual/alpha_stddev": jnp.mean(alpha_stddev),
                    "dual/temperature": jnp.mean(temperature),
                    "dual/penalty_temperature": jnp.mean(penalty_temperature),
                    "loss/unconst_policy_loss": unconst_policy_loss,
                    "loss/temperature_loss": loss_temperature,
                    "loss/critic_loss": critic_loss,
                    "kl/q_kl": jnp.mean(kl_nonparametric) / self.kl_epsilon,
                    "kl/action_penalty_kl": jnp.mean(penalty_kl_nonparametric) / self.kl_epsilon_penalty,
                    "Q_vals/mean_Q_pred": jnp.mean(q_value_pred),
                    "policy/mean_std_dev": jnp.mean(current_stddev)
                }

                # Some metrics seem to cost performance when logged, so they get only logged if necessary
                # TODO: When the jax logger works again, inspect why this is the case
                if self.logging_all_metrics:
                    metrics["loss/kl_penalty_loss"] = kl_penalty_loss
                    metrics["loss/alpha_loss"] = alpha_loss
                    metrics["kl/policy_mean_kl"] = jnp.mean(mean_kl_mean) / self.kl_epsilon_mean
                    metrics["kl/policy_stddev_kl"] = jnp.mean(mean_kl_stddev) / self.kl_epsilon_stddev

                return loss, (metrics)


            key, k1, k2 = jax.random.split(key, 3)

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, None, None), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))

            (loss, (metrics)), (agent_gradients, dual_gradients) = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)(train_state.agent_params, train_state.dual_params, train_state.agent_target_params, states, actions, rewards, dones, log_probs, k1, k2)

            agent_gradients_norm = optax.global_norm(agent_gradients)
            dual_gradients_norm = optax.global_norm(dual_gradients)

            agent_updates, agent_optimizer_state = self.agent_optimizer.update(agent_gradients, train_state.agent_optimizer_state, train_state.agent_params)
            dual_updates, dual_optimizer_state = self.dual_optimizer.update(dual_gradients, train_state.dual_optimizer_state, train_state.dual_params)

            agent_params = optax.apply_updates(train_state.agent_params, agent_updates)
            agent_target_params = optax.incremental_update(train_state.agent_params, train_state.agent_target_params, self.tau)

            dual_params = optax.apply_updates(train_state.dual_params, dual_updates)
            dual_params = DualParams(
                log_temperature=jnp.maximum(dual_params.log_temperature, self.min_log_temperature),
                log_alpha_mean=jnp.maximum(dual_params.log_alpha_mean, self.min_log_alpha),
                log_alpha_stddev=jnp.maximum(dual_params.log_alpha_stddev, self.min_log_alpha),
                log_penalty_temperature=jnp.maximum(dual_params.log_penalty_temperature, self.min_log_temperature)
            )

            train_state = TrainingState(
                agent_params=agent_params,
                agent_target_params=agent_target_params,
                dual_params=dual_params,
                agent_optimizer_state=agent_optimizer_state,
                dual_optimizer_state=dual_optimizer_state,
                steps=train_state.steps + 1
            )

            metrics["gradients/agent_gradients_norm"] = agent_gradients_norm
            metrics["gradients/dual_gradients_norm"] = dual_gradients_norm

            return train_state, metrics, key


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.trace_length, self.env.observation_space.shape, self.env.action_space.shape)

        state_stack = np.zeros((self.nr_envs, self.trace_length) + self.env.observation_space.shape, dtype=np.float32)
        action_stack = np.zeros((self.nr_envs, self.trace_length) + self.env.action_space.shape, dtype=np.float32)
        reward_stack = np.zeros((self.nr_envs, self.trace_length), dtype=np.float32)
        done_stack = np.zeros((self.nr_envs, self.trace_length), dtype=np.float32)
        log_prob_stack = np.zeros((self.nr_envs, self.trace_length), dtype=np.float32)

        saving_return_buffer = deque(maxlen=100)
        episode_info_buffer = deque(maxlen=self.logging_freq)
        acting_time_buffer = deque(maxlen=self.logging_freq)
        optimize_time_buffer = deque(maxlen=self.logging_freq)
        saving_time_buffer = deque(maxlen=self.logging_freq)
        fps_buffer = deque(maxlen=self.logging_freq)
        metrics_buffer = deque(maxlen=self.logging_freq)

        state = self.env.reset()

        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
                log_prob = get_log_prob(self.train_state.agent_params.policy_params, state, action)
            else:
                action, log_prob, self.key = get_action_and_log_prob(self.train_state.agent_params.policy_params, state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, done, info = self.env.step(jax.device_get(processed_action))
            
            global_step += self.nr_envs
            
            state_stack = np.roll(state_stack, shift=-1, axis=1)
            action_stack = np.roll(action_stack, shift=-1, axis=1)
            reward_stack = np.roll(reward_stack, shift=-1, axis=1)
            done_stack = np.roll(done_stack, shift=-1, axis=1)
            log_prob_stack = np.roll(log_prob_stack, shift=-1, axis=1)

            state_stack[:, -1] = state
            action_stack[:, -1] = action
            reward_stack[:, -1] = reward
            done_stack[:, -1] = done
            log_prob_stack[:, -1] = log_prob

            if global_step / self.nr_envs >= self.trace_length:
                replay_buffer.add(state_stack, action_stack, reward_stack, done_stack, log_prob_stack)

            state = next_state

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
                batch_states, batch_actions, batch_rewards, batch_dones, batch_log_probs = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                self.train_state, metrics, self.key = update(
                    self.train_state, batch_states, batch_actions, batch_rewards, batch_dones, batch_log_probs, self.key)
                metrics_buffer.append(metrics)

            optimize_end_time = time.time()
            optimize_time_buffer.append(optimize_end_time - acting_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
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
                self.log("lr/agent_learning_rate", self.train_state.agent_optimizer_state.hyperparams["learning_rate"].item(), global_step)
                self.log("lr/dual_learning_rate", self.train_state.dual_optimizer_state.hyperparams["learning_rate"].item(), global_step)
                if should_learning_start:
                    mean_metrics = {key: np.mean([metrics[key] for metrics in metrics_buffer]) for key in metrics_buffer[0].keys()}
                    for key, value in mean_metrics.items():
                        self.log(f"{key}", value, global_step)

                if self.track_console:
                    rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

                episode_info_buffer.clear()
                acting_time_buffer.clear()
                optimize_time_buffer.clear()
                saving_time_buffer.clear()
                fps_buffer.clear()
                metrics_buffer.clear()


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
        jax_file_path = self.save_path + "/model_best_jax_0"
        config_file_path = self.save_path + "/model_best_config_0"

        checkpoints.save_checkpoint(
            ckpt_dir=self.save_path,
            target={"train_state": self.train_state},
            step=0,
            prefix="model_best_jax_",
            overwrite=True
        )

        with open(config_file_path, "wb") as file:
            pickle.dump({"config_algorithm": self.config.algorithm}, file, pickle.HIGHEST_PROTOCOL)

        if self.track_wandb:
            wandb.save(jax_file_path, base_path=os.path.dirname(jax_file_path))
            wandb.save(config_file_path, base_path=os.path.dirname(config_file_path))
    
            
    def load(config, env, run_path, writer):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_name = splitted_path[-1]
        splitted_checkpoint_name = checkpoint_name.split("_")

        config_file_name = "_".join(splitted_checkpoint_name[:-2]) + "_config_" + splitted_checkpoint_name[-1]
        with open(f"{checkpoint_dir}/{config_file_name}", "rb") as file:
            config.algorithm = pickle.load(file)["config_algorithm"]
        model = MPO(config, env, run_path, writer)

        jax_file_name = "_".join(splitted_checkpoint_name[:-1]) + "_"
        step = int(splitted_checkpoint_name[-1])
        restored_train_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target={"train_state": model.train_state},
            step=step,
            prefix=jax_file_name
        )
        model.train_state = restored_train_state["train_state"]

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_params: flax.core.FrozenDict, state: np.ndarray):
            dist = self.policy.apply(policy_params, state)
            action = dist.mode()
            return self.get_processed_action(action)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                processed_action = get_action(self.train_state.agent_params.policy_params, state)
                state, reward, done, info = self.env.step(jax.device_get(processed_action))
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...
