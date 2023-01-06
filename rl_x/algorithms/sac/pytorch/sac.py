import os
import logging
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from rl_x.algorithms.sac.pytorch.policy import get_policy
from rl_x.algorithms.sac.pytorch.critic import get_critic
from rl_x.algorithms.sac.pytorch.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")


class SAC():
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
        self.logging_freq = config.algorithm.logging_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units

        self.device = torch.device("cuda" if config.algorithm.device == "gpu" and torch.cuda.is_available() else "cpu")
        rlx_logger.info(f"Using device: {self.device}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.policy = get_policy(config, env, self.device).to(self.device)
        self.q1 = get_critic(config, env).to(self.device)
        self.q2 = get_critic(config, env).to(self.device)
        self.q1_target = get_critic(config, env).to(self.device)
        self.q2_target = get_critic(config, env).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.learning_rate)

        if self.entropy_coef == "auto":
            if self.target_entropy == "auto":
                self.target_entropy = -torch.prod(torch.tensor(np.prod(env.get_single_action_space_shape()), dtype=torch.float32).to(self.device)).item()
            else:
                self.target_entropy = float(self.target_entropy)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.entropy_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.alpha = torch.tensor(float(self.entropy_coef)).to(self.device)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    
    def train(self):
        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.observation_space.shape, self.env.action_space.shape, self.device)

        saving_return_buffer = deque(maxlen=100)
        episode_info_buffer = deque(maxlen=self.logging_freq)
        acting_time_buffer = deque(maxlen=self.logging_freq)
        q_update_time_buffer = deque(maxlen=self.logging_freq)
        q_target_update_time_buffer = deque(maxlen=self.logging_freq)
        policy_update_time_buffer = deque(maxlen=self.logging_freq)
        entropy_update_time_buffer = deque(maxlen=self.logging_freq)
        saving_time_buffer = deque(maxlen=self.logging_freq)
        fps_buffer = deque(maxlen=self.logging_freq)
        q_loss_buffer = deque(maxlen=self.logging_freq)
        policy_loss_buffer = deque(maxlen=self.logging_freq)
        entropy_loss_buffer = deque(maxlen=self.logging_freq)
        entropy_buffer = deque(maxlen=self.logging_freq)
        alpha_buffer = deque(maxlen=self.logging_freq)

        state = self.env.reset()

        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            if global_step < self.learning_starts:
                action, processed_action = self.policy.get_random_actions(self.env, self.nr_envs)
            else:
                action, processed_action, _ = self.policy.get_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                action = action.detach().cpu().numpy()
                processed_action = processed_action.detach().cpu().numpy()
            
            next_state, reward, done, info = self.env.step(processed_action)
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
            should_update_q = should_learning_start and global_step % self.q_update_freq == 0
            should_update_q_target = should_learning_start and global_step % self.q_target_update_freq == 0
            should_update_policy = should_learning_start and global_step % self.policy_update_freq == 0
            should_update_entropy = should_learning_start and self.entropy_coef == "auto" and global_step % self.entropy_update_freq == 0
            should_try_to_save = should_learning_start and self.save_model and episode_infos 
            should_log = global_step % self.logging_freq == 0


            # Optimizing - Anneal learning rate
            learning_rate = self.learning_rate
            if self.anneal_learning_rate:
                fraction = 1 - (global_step / self.total_timesteps)
                learning_rate = fraction * self.learning_rate
                param_groups = self.policy_optimizer.param_groups + self.q_optimizer.param_groups
                if self.entropy_coef == "auto":
                    param_groups += self.entropy_optimizer.param_groups
                for param_group in param_groups:
                    param_group["lr"] = learning_rate

            
            # Optimizing - Prepare batches
            if should_update_q or should_update_policy or should_update_entropy:
                max_nr_batches_needed = max(should_update_q * self.q_update_steps, should_update_policy * self.policy_update_steps, should_update_entropy * self.entropy_update_steps)
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(self.batch_size, max_nr_batches_needed)


            # Optimizing - Q-functions
            if should_update_q:
                for i in range(self.q_update_steps):
                    with torch.no_grad():
                        next_actions, _, next_log_probs = self.policy.get_action(batch_next_states[i])
                        next_q1_target = self.q1_target(batch_next_states[i], next_actions)
                        next_q2_target = self.q2_target(batch_next_states[i], next_actions)
                        min_next_q_target = torch.minimum(next_q1_target, next_q2_target)
                        y = batch_rewards[i].reshape(-1, 1) + self.gamma * (1 - batch_dones[i].reshape(-1, 1)) * (min_next_q_target - self.alpha.detach() * next_log_probs)

                    q1 = self.q1(batch_states[i], batch_actions[i])
                    q2 = self.q2(batch_states[i], batch_actions[i])
                    q1_loss = F.mse_loss(q1, y)
                    q2_loss = F.mse_loss(q2, y)
                    q_loss = (q1_loss + q2_loss) / 2

                    self.q_optimizer.zero_grad()
                    q_loss.backward()
                    self.q_optimizer.step()

                    q_loss_buffer.append(q_loss.item())

            q_update_end_time = time.time()
            q_update_time_buffer.append(q_update_end_time - acting_end_time)


            # Optimizing - Q-function targets
            if should_update_q_target:
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            q_target_update_end_time = time.time()
            q_target_update_time_buffer.append(q_target_update_end_time - q_update_end_time)
            

            # Optimizing - Policy
            if should_update_policy:
                batch_entropies = []

                for i in range(self.policy_update_steps):
                    current_actions, _, current_log_probs = self.policy.get_action(batch_states[i])
                    
                    q1 = self.q1(batch_states[i], current_actions)
                    q2 = self.q2(batch_states[i], current_actions)
                    min_q = torch.minimum(q1, q2)
                    policy_loss = (self.alpha.detach() * current_log_probs - min_q).mean()  # sign switched compared to paper because paper uses gradient ascent

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    policy_loss_buffer.append(policy_loss.item())
                    alpha_buffer.append(self.alpha.item())

                    entropy = -current_log_probs.detach().mean()
                    batch_entropies.append(entropy)

            policy_update_end_time = time.time()
            policy_update_time_buffer.append(policy_update_end_time - q_target_update_end_time)
            

            # Optimizing - Entropy
            if should_update_entropy:
                for i in range(self.entropy_update_steps):
                    # Check if entropy was already calculated in policy optimization
                    if len(batch_entropies) >= i + 1:
                        entropy = batch_entropies[i]
                    else:
                        with torch.no_grad():
                            _, _, current_log_probs = self.policy.get_action(batch_actions[i])
                            entropy = -current_log_probs.detach().mean()

                    entropy_loss = self.alpha * (entropy - self.target_entropy)

                    self.entropy_optimizer.zero_grad()
                    entropy_loss.backward()
                    self.entropy_optimizer.step()

                    self.alpha = self.log_alpha.exp()

                    entropy_buffer.append(entropy.item())
                    entropy_loss_buffer.append(entropy_loss.item())

            entropy_update_end_time = time.time()
            entropy_update_time_buffer.append(entropy_update_end_time - policy_update_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            saving_time_buffer.append(saving_end_time - entropy_update_end_time)

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
                self.log("time/q_update_time", np.mean(q_update_time_buffer), global_step)
                self.log("time/q_target_update_time", np.mean(q_target_update_time_buffer), global_step)
                self.log("time/policy_update_time", np.mean(policy_update_time_buffer), global_step)
                self.log("time/entropy_update_time", np.mean(entropy_update_time_buffer), global_step)
                self.log("time/saving_time", np.mean(saving_time_buffer), global_step)
                self.log("train/learning_rate", learning_rate, global_step)
                self.log("train/q_loss", self.get_buffer_mean(q_loss_buffer), global_step)
                self.log("train/policy_loss", self.get_buffer_mean(policy_loss_buffer), global_step)
                self.log("train/entropy_loss", self.get_buffer_mean(entropy_loss_buffer), global_step)
                self.log("train/entropy", self.get_buffer_mean(entropy_buffer), global_step)
                self.log("train/alpha", self.get_buffer_mean(alpha_buffer), global_step)

                if self.track_console:
                    rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

                episode_info_buffer.clear()
                acting_time_buffer.clear()
                q_update_time_buffer.clear()
                q_target_update_time_buffer.clear()
                policy_update_time_buffer.clear()
                entropy_update_time_buffer.clear()
                saving_time_buffer.clear()
                fps_buffer.clear()
                q_loss_buffer.clear()
                policy_loss_buffer.clear()
                entropy_loss_buffer.clear()
                entropy_buffer.clear()
                alpha_buffer.clear()


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
        file_path = self.save_path + "/model_best.pt"
        save_dict = {
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
        }
        if self.entropy_coef == "auto":
            save_dict["log_alpha"] = self.log_alpha
            save_dict["entropy_optimizer_state_dict"] = self.entropy_optimizer.state_dict()
        torch.save(save_dict, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, env, run_path, writer):
        checkpoint = torch.load(config.runner.load_model)
        config.algorithm = checkpoint["config_algorithm"]
        model = SAC(config, env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.q1.load_state_dict(checkpoint["q1_state_dict"])
        model.q2.load_state_dict(checkpoint["q2_state_dict"])
        model.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        model.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        if config.algorithm.entropy_coef == "auto":
            model.log_alpha = checkpoint["log_alpha"]
            model.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer_state_dict"])

        return model

    
    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                processed_action = self.policy.get_deterministic_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                state, reward, done, info = self.env.step(processed_action.cpu().numpy())
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")


    def set_train_mode(self):
        self.policy.train()
        self.q1.train()
        self.q2.train()
        self.q1_target.train()
        self.q2_target.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
