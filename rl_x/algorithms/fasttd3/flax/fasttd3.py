import os
import shutil
import json
import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.fasttd3.flax.general_properties import GeneralProperties
from rl_x.algorithms.fasttd3.flax.policy import get_policy
from rl_x.algorithms.fasttd3.flax.critic import get_critic
from rl_x.algorithms.fasttd3.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.fasttd3.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class FastTD3:
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
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.weight_decay = config.algorithm.weight_decay
        self.batch_size = config.algorithm.batch_size
        self.buffer_size_per_env = config.algorithm.buffer_size_per_env
        self.learning_starts = config.algorithm.learning_starts
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.nr_atoms = config.algorithm.nr_atoms
        self.n_steps = config.algorithm.n_steps
        self.noise_std_min = config.algorithm.noise_std_min
        self.noise_std_max = config.algorithm.noise_std_max
        self.smoothing_epsilon = config.algorithm.smoothing_epsilon
        self.smoothing_clip_value = config.algorithm.smoothing_clip_value
        self.nr_critic_updates_per_policy_update = config.algorithm.nr_critic_updates_per_policy_update
        self.nr_policy_updates_per_step = config.algorithm.nr_policy_updates_per_step
        self.clipped_double_q_learning = config.algorithm.clipped_double_q_learning
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.action_clipping_and_rescaling = config.algorithm.action_clipping_and_rescaling
        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.normalizer_epsilon = config.algorithm.normalizer_epsilon
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.save_frequency = config.algorithm.save_frequency
        self.horizon = self.train_env.horizon

        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("The logging frequency must be a multiple of the number of environments.")
        
        if self.save_frequency != -1 and self.save_frequency % self.nr_envs != 0:
            raise ValueError("The save frequency must be a multiple of the number of environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            step = (count * self.nr_envs) - (self.learning_starts * self.nr_envs)
            total_steps = self.total_timesteps - (self.learning_starts * self.nr_envs)
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.train_env.single_observation_space.sample()])
        action = jnp.array([self.train_env.single_action_space.sample()])

        if self.max_grad_norm != -1.0:
            policy_tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(learning_rate=self.policy_learning_rate, weight_decay=self.weight_decay),
            )
        else:
            policy_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=self.policy_learning_rate, weight_decay=self.weight_decay)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            tx=policy_tx
        )

        if self.max_grad_norm != -1.0:
            critic_tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(learning_rate=self.q_learning_rate, weight_decay=self.weight_decay),
            )
        else:
            critic_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=self.q_learning_rate, weight_decay=self.weight_decay)
        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, action),
            target_params=self.critic.init(critic_key, state, action),
            tx=critic_tx
        )

        if self.enable_observation_normalization:
            self.observation_normalizer_state = {
                "running_mean": np.zeros((1, state.shape[-1]), dtype=np.float32),
                "running_var": np.ones((1, state.shape[-1]), dtype=np.float32),
                "running_std_dev": np.ones((1, state.shape[-1]), dtype=np.float32),
                "count": 0
            }
        else:
            self.observation_normalizer_state = None

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def _normalize_observations(self, observations, update=False, no_normalize_after_update=True):
        if not self.enable_observation_normalization:
            return observations

        if update:
            batch_mean = np.mean(observations, axis=0, keepdims=True)
            batch_var = np.var(observations, axis=0, keepdims=True)
            batch_count = observations.shape[0]

            new_count = self.observation_normalizer_state["count"] + batch_count
            delta = batch_mean - self.observation_normalizer_state["running_mean"]
            self.observation_normalizer_state["running_mean"] += delta * batch_count / new_count
            delta2 = batch_mean - self.observation_normalizer_state["running_mean"]
            m_a = self.observation_normalizer_state["running_var"] * self.observation_normalizer_state["count"]
            m_b = batch_var * batch_count
            m2 = m_a + m_b + np.square(delta2) * self.observation_normalizer_state["count"] * batch_count / new_count
            self.observation_normalizer_state["running_var"] = m2 / new_count
            self.observation_normalizer_state["running_std_dev"] = np.sqrt(self.observation_normalizer_state["running_var"])
            self.observation_normalizer_state["count"] = new_count

            if no_normalize_after_update:
                return

        return (observations - self.observation_normalizer_state["running_mean"]) / (self.observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)


    def train(self):
        @jax.jit
        def update_critic(policy_params, critic_state, normalized_states, normalized_next_states, actions, rewards, dones, truncations, all_effective_n_steps, key):
            def loss_fn(critic_params, normalized_state, normalized_next_state, action, reward, done, truncated, effective_n_steps, key1):
                clipped_noise = jnp.clip(jax.random.normal(key1, action.shape) * self.smoothing_epsilon, -self.smoothing_clip_value, self.smoothing_clip_value)
                next_action = jnp.clip(self.policy.apply(policy_params, normalized_next_state) + clipped_noise, -1.0, 1.0)

                delta_z = (self.v_max - self.v_min) / (self.nr_atoms - 1)
                q_support = jnp.linspace(self.v_min, self.v_max, self.nr_atoms)
                bootstrap = 1.0 - (done * (1.0 - truncated))
                discount = (self.gamma ** effective_n_steps) * bootstrap
                target_z = jnp.clip(reward + discount * q_support, self.v_min, self.v_max)
                b = (target_z - self.v_min) / delta_z
                l = jnp.floor(b).astype(jnp.int32)
                u = jnp.ceil(b).astype(jnp.int32)

                is_int = (u == l)
                l_mask = is_int & (l > 0)
                u_mask = is_int & (l == 0)

                l = jnp.where(l_mask, l - 1, l)
                u = jnp.where(u_mask, u + 1, u)

                next_dist = jax.nn.softmax(self.critic.apply(critic_state.target_params, normalized_next_state, next_action))
                proj_dist = jnp.zeros_like(next_dist)
                wt_l = (u.astype(jnp.float32) - b)
                wt_u = (b - l.astype(jnp.float32))

                n_critics = next_dist.shape[0]
                critic_idxs = jnp.arange(n_critics)[:, None]
                critic_idxs = jnp.repeat(critic_idxs, self.nr_atoms, axis=1)  
                l_idxs = jnp.repeat(l[None, :], n_critics, axis=0)
                u_idxs = jnp.repeat(u[None, :], n_critics, axis=0)

                proj_dist = proj_dist.at[(critic_idxs, l_idxs)].add(next_dist * wt_l)
                proj_dist = proj_dist.at[(critic_idxs, u_idxs)].add(next_dist * wt_u)

                qf_next_target_value = jnp.sum(proj_dist * q_support, axis=1)  # (2,)

                if self.clipped_double_q_learning:
                    qf_next_target_dist = jnp.where(qf_next_target_value[0] < qf_next_target_value[1], proj_dist[0], proj_dist[1])
                    qf1_next_target_dist = qf_next_target_dist
                    qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist = proj_dist[0]
                    qf2_next_target_dist = proj_dist[1]

                current_q = self.critic.apply(critic_params, normalized_state, action)

                q1_loss = -jnp.sum(qf1_next_target_dist * jax.nn.log_softmax(current_q[0]), axis=-1)
                q2_loss = -jnp.sum(qf2_next_target_dist * jax.nn.log_softmax(current_q[1]), axis=-1)

                loss = q1_loss + q2_loss

                metrics = {
                    "loss/q_loss": loss,
                    "q/q_max": jnp.max(qf_next_target_value),
                    "q/q_min": jnp.min(qf_next_target_value),
                }

                return loss, metrics

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_loss_fn = lambda *a, **k: jax.tree_util.tree_map(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_loss_fn, argnums=(0,), has_aux=True)

            key, subkeys = jax.random.split(key, 2)
            per_sample_keys = jax.random.split(subkeys, normalized_states.shape[0])

            (loss, metrics), (critic_grads,) = grad_loss_fn(critic_state.params, normalized_states, normalized_next_states, actions, rewards, dones, truncations, all_effective_n_steps, per_sample_keys)
            
            critic_state = critic_state.apply_gradients(grads=critic_grads)
            
            critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))

            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_grads)

            return critic_state, metrics, key


        @jax.jit
        def update_policy(policy_state, critic_state, states):
            def loss_fn(policy_params, normalized_state):
                action = self.policy.apply(policy_params, normalized_state)

                q_values = self.critic.apply(critic_state.params, normalized_state, action)
                q_values = jnp.sum(jax.nn.softmax(q_values) * jnp.linspace(self.v_min, self.v_max, self.nr_atoms), axis=-1)
                if self.clipped_double_q_learning:
                    processed_q_value = jnp.min(q_values, axis=0)
                else:
                    processed_q_value = jnp.mean(q_values, axis=0)
                
                loss = -jnp.mean(processed_q_value)
                
                metrics = {"loss/policy_loss": loss}
                
                return loss, metrics

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0))
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_loss_fn = lambda *a, **k: jax.tree_util.tree_map(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_loss_fn, argnums=(0,), has_aux=True)

            (loss, metrics), (policy_grads,) = grad_loss_fn(policy_state.params, states)
            policy_state = policy_state.apply_gradients(grads=policy_grads)

            metrics["lr/learning_rate"] = policy_state.opt_state.hyperparams["learning_rate"]
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_grads)

            return policy_state, metrics


        self.set_train_mode()

        replay_buffer = ReplayBuffer(self.buffer_size_per_env, self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.n_steps, self.gamma, self.rng)

        state, _ = self.train_env.reset()
        self.key, subkey = jax.random.split(self.key)
        noise_scales = jax.random.uniform(subkey, (self.nr_envs, 1), minval=self.noise_std_min, maxval=self.noise_std_max)
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None
        
        while global_step < self.total_timesteps:
            start_time = time.time()
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)


            # Acting
            dones_this_rollout = 0
            normalized_state = self._normalize_observations(state, update=False)
            action = self.policy.apply(self.policy_state.params, normalized_state)
            self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
            action = action + jax.random.normal(subkey1, action.shape) * noise_scales
            processed_action = self.get_processed_action(action)
            next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                    dones_this_rollout += 1
            for key, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)

            replay_buffer.add(state, actual_next_state, action, reward, done, truncated)

            noise_scales = jnp.where(
                done[:, None],
                jax.random.uniform(subkey2, (self.nr_envs, 1), minval=self.noise_std_min, maxval=self.noise_std_max),
                noise_scales
            )

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts * self.nr_envs
            should_optimize = should_learning_start
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0 and self.save_frequency != -1 and global_step % self.save_frequency == 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing
            if should_optimize:
                total_critic_updates = self.nr_policy_updates_per_step * self.nr_critic_updates_per_policy_update
                total_batch_size = total_critic_updates * self.batch_size
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncations, batch_effective_n_steps = replay_buffer.sample(total_batch_size)

                if self.enable_observation_normalization:
                    combined_states = np.concatenate([batch_states, batch_next_states], axis=0)
                    self._normalize_observations(combined_states, update=True, no_normalize_after_update=True)
                normalized_states = self._normalize_observations(batch_states, update=False)
                normalized_next_states = self._normalize_observations(batch_next_states, update=False)

                normalized_states = normalized_states.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                normalized_next_states = normalized_next_states.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                batch_actions = batch_actions.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                batch_rewards = batch_rewards.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_dones = batch_dones.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_truncations = batch_truncations.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_effective_n_steps = batch_effective_n_steps.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)

                for i in range(self.nr_policy_updates_per_step):
                    for j in range(self.nr_critic_updates_per_policy_update):
                        self.key, subkey = jax.random.split(self.key)
                        self.critic_state, optimization_metrics, self.key = update_critic(
                            self.policy_state.params,
                            self.critic_state,
                            normalized_states[i, j],
                            normalized_next_states[i, j],
                            batch_actions[i, j],
                            batch_rewards[i, j],
                            batch_dones[i, j],
                            batch_truncations[i, j],
                            batch_effective_n_steps[i, j],
                            subkey,
                        )
                        for key, value in optimization_metrics.items():
                            optimization_metrics_collection.setdefault(key, []).append(value)
                        nr_updates += 1

                    self.policy_state, optimization_metrics = update_policy(self.policy_state, self.critic_state, normalized_states[i, -1])
                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                for _ in range(self.horizon):
                    eval_normalized_state = self._normalize_observations(eval_state, update=False)
                    eval_action = self.policy.apply(self.policy_state.params, eval_normalized_state)
                    eval_processed_action = self.get_processed_action(eval_action)
                    eval_state, _, _, _, eval_info = self.eval_env.step(jax.device_get(eval_processed_action))
                    eval_logging_info = self.eval_env.get_logging_info_dict(eval_info)
                    if "episode_return" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_return", []).extend(eval_logging_info["episode_return"])
                    if "episode_length" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_length", []).extend(eval_logging_info["episode_length"])
                self.set_train_mode()

            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                self.save()

            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)


            # Logging
            if should_log:
                self.start_logging(global_step)

                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_critic_updates"] = nr_updates
                steps_metrics["steps/nr_policy_updates"] = nr_updates // self.nr_critic_updates_per_policy_update
                steps_metrics["steps/nr_episodes"] = nr_episodes

                rollout_info_metrics = {}
                env_info_metrics = {}
                if step_info_collection:
                    info_names = list(step_info_collection.keys())
                    for info_name in info_names:
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(step_info_collection[info_name])
                        if mean_value == mean_value:  # Check if mean_value is NaN
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}

                self.end_logging()
            
            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time


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


    def save(self):
        checkpoint = {
            "policy": self.policy_state,
            "critic": self.critic_state,
            "observation_normalizer": self.observation_normalizer_state
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
        model = FastTD3(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "observation_normalizer": model.observation_normalizer_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.observation_normalizer_state = checkpoint.get("observation_normalizer", model.observation_normalizer_state)

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                normalized_state = self._normalize_observations(state, update=False)
                action = self.policy.apply(self.policy_state.params, normalized_state)
                processed_action = self.get_processed_action(action)
                state, reward, terminated, truncated, _ = self.eval_env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
