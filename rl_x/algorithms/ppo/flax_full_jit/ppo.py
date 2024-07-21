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
        self.evaluation_active = config.algorithm.evaluation_frequency != -1
        if config.algorithm.evaluation_frequency == -1:
            self.evaluation_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_iterations = self.total_timesteps // self.evaluation_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_frequency // self.batch_size
        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape

        if self.evaluation_frequency % self.batch_size != 0:
            raise ValueError("Evaluation frequency must be a multiple of batch size")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.env)
        self.critic = get_critic(self.config, self.env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.env.reset(reset_key)

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

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)

 
    def train(self):
        def jitable_train_function(key):
            key, subkey = jax.random.split(key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.env.reset(reset_keys)

            def multi_learning_and_eval_iteration(multi_learning_and_eval_iteration_carry, multi_learnning_iteration_step):
                policy_state, critic_state, env_state, key = multi_learning_and_eval_iteration_carry

                def learning_iteration(learning_iteration_carry, learnning_iteration_step):
                    policy_state, critic_state, env_state, key = learning_iteration_carry

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, env_state, key = single_rollout_carry

                        key, subkey = jax.random.split(key)
                        action_mean, action_logstd = self.policy.apply(policy_state.params, env_state.next_observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                        processed_action = self.get_processed_action(action)
                        value = self.critic.apply(critic_state.params, env_state.next_observation).squeeze(-1)

                        env_state = self.env.step(env_state, processed_action)
                        transition = (env_state.next_observation, env_state.actual_next_observation, action, env_state.reward, value, env_state.terminated, log_prob, env_state.info)

                        return (policy_state, critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, learning_iteration_carry, None, self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, log_probs, infos = batch


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

                    advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values, terminations)


                    # Optimizing
                    def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                        # Policy loss
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
                        
                        # Critic loss
                        new_value = self.critic.apply(critic_params, state_b)
                        critic_loss = 0.5 * (new_value - return_b) ** 2

                        # Combine losses
                        loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

                        # Create metrics
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

                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
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
                            minibatch_advantages
                        )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                        carry = (policy_state, critic_state)

                        return carry, (metrics)
                    
                    init_carry = (policy_state, critic_state)
                    carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state = carry

                    optimization_metrics = {key: jnp.mean(optimization_metrics[key]) for key in optimization_metrics}
                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))


                    # Saving
                    if self.save_model:
                        def save_with_check(infos):
                            mean_return = np.mean(infos["rollout/episode_return"])
                            if mean_return > self.best_mean_return:
                                self.best_mean_return = mean_return
                                self.save()
                        jax.debug.callback(save_with_check, infos)


                    # Logging
                    steps_metrics = {
                        "steps/nr_env_steps": (multi_learnning_iteration_step + 1) * (learnning_iteration_step + 1) * self.nr_steps * self.nr_envs,
                        "steps/nr_updates": (multi_learnning_iteration_step + 1) * (learnning_iteration_step + 1) * self.nr_epochs * self.nr_minibatches,
                    }

                    combined_metrics = {**infos, **steps_metrics, **optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(metrics):
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time))
                        self.last_time = current_time
                        global_step = int(metrics["steps/nr_env_steps"])
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(callback, combined_metrics)
                    
                    return (policy_state, critic_state, env_state, key), None
                    
                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, env_state, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, env_state, key = learning_iteration_carry


                # Evaluating
                if self.evaluation_active:
                    def evaluate_single(policy_state, env_state):
                        def single_deterministic_rollout(single_rollout_carry):
                            policy_state, env_state, _, episode_return, episode_length = single_rollout_carry
                            action_mean, _ = self.policy.apply(policy_state.params, env_state.next_observation)
                            processed_action = self.get_processed_action(action_mean)
                            env_state = self.env._step(env_state, processed_action)
                            done = env_state.terminated | env_state.truncated
                            return (policy_state, env_state, done, episode_return + env_state.reward, episode_length + 1)

                        single_rollout_carry = jax.lax.while_loop(
                            lambda carry: jnp.logical_not(carry[2]),
                            single_deterministic_rollout,
                            (policy_state, env_state, False, 0.0, 0)
                        )
                        _, _, _, episode_return, episode_length = single_rollout_carry
                        return {"eval/episode_return": episode_return, "eval/episode_length": episode_length}
                    
                    key, subkey = jax.random.split(key)
                    reset_keys = jax.random.split(subkey, self.evaluation_episodes)
                    eval_env_state = self.env.reset(reset_keys)
                    eval_infos = jax.vmap(evaluate_single, in_axes=(None, 0))(policy_state, eval_env_state)
                    eval_infos = tree.map_structure(lambda x: jnp.mean(x), eval_infos)

                    def callback(metrics_and_global_step):
                        metrics, global_step = metrics_and_global_step
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    global_step = (multi_learnning_iteration_step + 1) * self.evaluation_frequency
                    jax.debug.callback(callback, (eval_infos, global_step))
            
                return (policy_state, critic_state, env_state, key), None

            jax.lax.scan(multi_learning_and_eval_iteration, (self.policy_state, self.critic_state, env_state, key), jnp.arange(self.nr_multi_learning_and_eval_iterations))
            

        self.key, subkey = jax.random.split(self.key)
        train_function = jax.jit(jitable_train_function)
        self.last_time = time.time()
        self.start_time = self.last_time
        jax.block_until_ready(train_function(subkey))
        rlx_logger.info(f"time: {time.time() - self.start_time:.2f} s")
    

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
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": self.policy_state,
            "critic": self.critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        shutil.make_archive(f"{self.save_path}/{self.best_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.best_model_file_name}.zip", f"{self.save_path}/{self.best_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)
    

    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1]) if len(splitted_path) > 1 else "."
        checkpoint_file_name = splitted_path[-1]

        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"
        jax_model_file_name = [f for f in os.listdir(checkpoint_dir) if f.endswith(".model")][0]

        check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=jax_model_file_name)
        checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)

        loaded_algorithm_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = PPO(config, env, run_path, writer)

        target = {
            "config_algorithm": config.algorithm.to_dict(),
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model


    def test(self, episodes):
        @jax.jit
        def run_single(policy_state, env_state):
            def deterministic_rollout(single_rollout_carry):
                policy_state, env_state, _, episode_return = single_rollout_carry
                action_mean, _ = self.policy.apply(policy_state.params, env_state.next_observation)
                processed_action = self.get_processed_action(action_mean)
                env_state = self.env.step(env_state, processed_action)
                done = env_state.terminated | env_state.truncated
                return (policy_state, env_state, done[0], episode_return + env_state.reward[0])

            single_rollout_carry = jax.lax.while_loop(
                lambda carry: jnp.logical_not(carry[2]),
                deterministic_rollout,
                (policy_state, env_state, False, 0.0)
            )
            _, _, _, episode_return = single_rollout_carry
            return episode_return
        
        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, 1)
        env_state = self.env.reset(reset_keys)
        for i in range(episodes):
            episode_return = run_single(self.policy_state, env_state)
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def general_properties():
        return GeneralProperties
