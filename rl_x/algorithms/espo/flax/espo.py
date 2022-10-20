import os
import random
import logging
import pickle
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

from rl_x.algorithms.espo.flax.actor import get_actor
from rl_x.algorithms.espo.flax.critic import get_critic
from rl_x.algorithms.espo.flax.agent_params import AgentParams
from rl_x.algorithms.espo.flax.storage import Storage

rlx_logger = logging.getLogger("rl_x")


class ESPO:
    def __init__(self, config, env, writer):  
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
        self.nr_steps = config.algorithm.nr_steps
        self.max_epochs = config.algorithm.max_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.max_ratio_delta = config.algorithm.max_ratio_delta
        self.ent_coef = config.algorithm.ent_coef
        self.vf_coef = config.algorithm.vf_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.batch_size = config.algorithm.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size

        if config.algorithm.delta_calc_operator == "mean":
            self.delta_calc_operator = jnp.mean
        elif config.algorithm.delta_calc_operator == "median":
            self.delta_calc_operator = jnp.median
        else:
            raise ValueError("Unknown delta_calc_operator")

        if config.algorithm.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, actor_key, critic_key = jax.random.split(self.key, 3)

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.max_epochs)) / self.nr_updates
            return self.learning_rate * fraction
        
        self.actor, self.get_processed_action = get_actor(config, env)
        self.critic = get_critic(config, env)

        self.first_state = env.reset()
        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.train_state = TrainState.create(
            apply_fn=None,
            params=AgentParams(
                self.actor.init(actor_key, self.first_state),
                self.critic.init(critic_key, self.first_state),
            ),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            ),
        )
        
        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    
    def train(self):
        @jax.jit
        def get_action_and_value(params: flax.core.FrozenDict, state: np.ndarray, storage: Storage, step: int, key: jax.random.PRNGKey):
            action_mean, action_logstd = jax.lax.stop_gradient(self.actor.apply(params.actor_params, state))
            action_std = jnp.exp(action_logstd)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            value = self.critic.apply(params.critic_params, state)
            storage = storage.replace(
                states=storage.states.at[step].set(state),
                actions=storage.actions.at[step].set(action),
                log_probs=storage.log_probs.at[step].set(log_prob.sum(1)),
                values=storage.values.at[step].set(value.reshape(-1))
            )
            processed_action = self.get_processed_action(action)
            return processed_action, storage, key
        

        @jax.jit
        def get_action_and_value2(params: flax.core.FrozenDict, state: np.ndarray, action: np.ndarray):
            action_mean, action_logstd = self.actor.apply(params.actor_params, state)
            action_std = jnp.exp(action_logstd)
            logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
            value = self.critic.apply(params.critic_params, state)
            return logprob.sum(1), entropy, value.reshape(-1)
        

        @jax.jit
        def calculate_gae_advantages(params: flax.core.FrozenDict, next_states: jnp.array, rewards: np.ndarray, dones: np.ndarray, storage: Storage):
            next_values = jax.lax.stop_gradient(self.critic.apply(params.critic_params, next_states)).squeeze()
            delta = rewards + self.gamma * next_values * (1.0 - dones) - storage.values
            advantages = [delta[-1]]
            for t in jnp.arange(self.nr_steps - 2, -1, -1):
                advantages.insert(0, delta[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * advantages[0])
            advantages = jnp.array(advantages)
            storage = storage.replace(
                advantages=advantages,
                returns=advantages + storage.values
            )
            return storage
        

        @jax.jit
        def update(train_state: TrainState, storage: Storage, minibatch_indices: jnp.array):
            batch_states = storage.states.reshape((-1,) + self.os_shape)
            batch_actions = storage.actions.reshape((-1,) + self.as_shape)
            batch_advantages = storage.advantages.reshape(-1)
            batch_returns = storage.returns.reshape(-1)
            batch_log_probs = storage.log_probs.reshape(-1)

            def ppo_loss(params, minibatch_states, minibatch_actions, minibatch_log_probs, minibatch_advantages, minibatch_returns):
                new_log_prob, entropy, new_value = get_action_and_value2(params, minibatch_states, minibatch_actions)
                logratio = new_log_prob - minibatch_log_probs
                ratio = jnp.exp(logratio)
                approx_kl_div = jax.lax.stop_gradient(((ratio - 1) - logratio).mean())

                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                pg_loss = (-minibatch_advantages * ratio).mean()
                v_loss = (0.5 * (new_value - minibatch_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

                return loss, (pg_loss, v_loss, entropy_loss, approx_kl_div, ratio)

            ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

            (loss, (pg_loss, v_loss, entropy_loss, approx_kl_div, ratio)), grads = ppo_loss_grad_fn(train_state.params,
                batch_states[minibatch_indices],
                batch_actions[minibatch_indices],
                batch_log_probs[minibatch_indices],
                batch_advantages[minibatch_indices],
                batch_returns[minibatch_indices]
            )
            train_state = train_state.apply_gradients(grads=grads)

            return train_state, ratio, loss, pg_loss, v_loss, entropy_loss, approx_kl_div


        @jax.jit
        def calculate_explained_variance(values: jnp.array, returns: jnp.array):
            return 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
        

        storage = Storage(
            states=jnp.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=jnp.zeros((self.nr_steps, self.nr_envs) + self.as_shape),
            log_probs=jnp.zeros((self.nr_steps, self.nr_envs)),
            values=jnp.zeros((self.nr_steps, self.nr_envs)),
            advantages=jnp.zeros((self.nr_steps, self.nr_envs)),
            returns=jnp.zeros((self.nr_steps, self.nr_envs)),
        )
        rewards = np.zeros((self.nr_steps, self.nr_envs))
        dones = np.zeros((self.nr_steps, self.nr_envs))
        next_states = np.zeros((self.nr_steps, self.nr_envs) + self.os_shape)

        state = self.first_state
        saving_return_buffer = deque(maxlen=100)
        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            episode_info_buffer = deque(maxlen=100)
            for step in range(self.nr_steps):
                processed_action, storage, self.key = get_action_and_value(self.train_state.params, state, storage, step, self.key)
                next_state, reward, done, info = self.env.step(jax.device_get(processed_action))
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        maybe_terminal_observation = self.env.get_terminal_observation(info, i)
                        if maybe_terminal_observation is not None:
                            actual_next_state[i] = maybe_terminal_observation

                next_states[step] = actual_next_state
                rewards[step] = reward
                dones[step] = done
                state = next_state
                global_step += self.nr_envs

                episode_info_buffer.extend(self.env.get_episode_infos(info))
                if len(episode_info_buffer) > 0:
                    ep_info_returns = [ep_info["r"] for ep_info in episode_info_buffer]
                    saving_return_buffer.extend(ep_info_returns)
            
            acting_end_time = time.time()


            # Calculating advantages and returns
            storage = calculate_gae_advantages(self.train_state.params, next_states, rewards, dones, storage)
            
            calc_adv_return_end_time = time.time()


            # Optimizing
            pg_losses = []
            value_losses = []
            entropy_losses = []
            loss_losses = []
            ratio_deltas = []

            for epoch in range(self.max_epochs):
                self.key, subkey = jax.random.split(self.key)
                minibatch_indices = jax.random.choice(subkey, self.batch_size, shape=(self.minibatch_size,), replace=False)
                self.train_state, ratio, loss, pg_loss, v_loss, entropy_loss, approx_kl_div = update(self.train_state, storage, minibatch_indices)
                
                episode_ratio_delta = self.delta_calc_operator(jnp.abs(ratio - 1))

                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                loss_losses.append(loss.item())
                ratio_deltas.append(episode_ratio_delta.item())

                if episode_ratio_delta > self.max_ratio_delta:
                    break

            explained_var = calculate_explained_variance(storage.values, storage.returns)

            optimizing_end_time = time.time()
            

            # Saving
            # Only save when the total return buffer (over multiple updates) isn't empty
            # Also only save when the episode info buffer isn't empty -> there were finished episodes this update
            if self.save_model and saving_return_buffer and episode_info_buffer:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()


            # Logging
            if self.track_tb:
                if len(episode_info_buffer) > 0:
                    self.writer.add_scalar("rollout/ep_rew_mean", np.mean(ep_info_returns), global_step)
                    self.writer.add_scalar("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in episode_info_buffer]), global_step)
                    names = list(episode_info_buffer[0].keys())
                    for name in names:
                        if name != "r" and name != "l" and name != "t":
                            self.writer.add_scalar(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info.keys()]), global_step)
                self.writer.add_scalar("time/fps", int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time)), global_step)
                self.writer.add_scalar("time/acting_time", acting_end_time - start_time, global_step)
                self.writer.add_scalar("time/calc_advantages_and_return_time", calc_adv_return_end_time - acting_end_time, global_step)
                self.writer.add_scalar("time/optimizing_time", optimizing_end_time - calc_adv_return_end_time, global_step)
                self.writer.add_scalar("time/saving_time", saving_end_time - optimizing_end_time, global_step)
                self.writer.add_scalar("train/learning_rate", self.train_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
                self.writer.add_scalar("train/epochs", epoch + 1, global_step)
                self.writer.add_scalar("train/ratio_delta", np.mean(ratio_deltas), global_step)
                self.writer.add_scalar("train/last_approx_kl", approx_kl_div.item(), global_step)
                self.writer.add_scalar("train/policy_gradient_loss", np.mean(pg_losses), global_step)
                self.writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
                self.writer.add_scalar("train/entropy_loss", np.mean(entropy_losses), global_step)
                self.writer.add_scalar("train/loss", np.mean(loss_losses), global_step)
                self.writer.add_scalar("train/std", np.mean(np.asarray(jnp.exp(self.train_state.params.actor_params["params"]["actor_logstd"]))), global_step)
                self.writer.add_scalar("train/explained_variance", explained_var.item(), global_step)

            rlx_logger.info(f"Step: {global_step}")

    
    def save(self):
        jax_file_path = self.save_path + "/model_best_jax_0"
        config_file_path = self.save_path + "/model_best_config_0"

        checkpoints.save_checkpoint(
            ckpt_dir=self.save_path,
            target=self.train_state,
            step=0,
            prefix="model_best_jax_",
            overwrite=True
        )

        with open(config_file_path, "wb") as file:
            pickle.dump({"config_algorithm": self.config.algorithm}, file, pickle.HIGHEST_PROTOCOL)

        if self.track_wandb:
            wandb.save(jax_file_path, base_path=os.path.dirname(jax_file_path))
            wandb.save(config_file_path, base_path=os.path.dirname(config_file_path))
    

    def load(config, env, writer):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_name = splitted_path[-1]
        splitted_checkpoint_name = checkpoint_name.split("_")

        config_file_name = "_".join(splitted_checkpoint_name[:-2]) + "_config_" + splitted_checkpoint_name[-1]
        with open(f"{checkpoint_dir}/{config_file_name}", "rb") as file:
            config.algorithm = pickle.load(file)["config_algorithm"]
        model = ESPO(config, env, writer)

        jax_file_name = "_".join(splitted_checkpoint_name[:-1]) + "_"
        step = int(splitted_checkpoint_name[-1])
        restored_train_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=model.train_state,
            step=step,
            prefix=jax_file_name
        )
        model.train_state = restored_train_state

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(params: flax.core.FrozenDict, state: np.ndarray):
            action_mean, action_logstd = jax.lax.stop_gradient(self.actor.apply(params.actor_params, state))
            return self.get_processed_action(action_mean)
        
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                processed_action = get_action(self.train_state.params, state)
                state, reward, done, info = self.env.step(jax.device_get(processed_action))
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")
