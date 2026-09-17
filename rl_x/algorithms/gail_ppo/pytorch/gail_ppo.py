import logging
import math
import os
import time
import numpy as np
import torch
import wandb

from rl_x.environments.data_interface_type import DataInterfaceType
from rl_x.algorithms.gail_ppo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.gail_ppo.pytorch.policy import Policy
from rl_x.algorithms.gail_ppo.pytorch.critic import Critic
from rl_x.algorithms.gail_ppo.pytorch.data_utils import prepare_expert_data
from rl_x.algorithms.gail_ppo.pytorch.discriminator import Discriminator

rlx_logger = logging.getLogger("rl_x")


class GAIL_PPO:
    def __init__(self, config, train_env, eval_env, run_path, writer):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer
        self.save_path = os.path.join(run_path, "models")
        self.is_torch_data_interface = train_env.general_properties.data_interface_type == DataInterfaceType.TORCH
        device = "cuda" if config.algorithm.device == "gpu" and torch.cuda.is_available() else "mps" if config.algorithm.device == "mps" and torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        torch.manual_seed(config.environment.seed)
        self.batch_size = config.algorithm.nr_steps * config.environment.nr_envs
        self.nr_updates = int(config.algorithm.total_timesteps // self.batch_size)
        self.nr_minibatches = self.batch_size // config.algorithm.minibatch_size
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        if self.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.nr_updates * self.batch_size
        if config.runner.mode == "train" and (self.nr_minibatches < 1 or self.nr_updates < 1):
            raise ValueError("Training requires at least one full rollout and minibatch")
        if config.runner.mode == "train" and self.evaluation_and_save_frequency % self.batch_size:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")
        if config.algorithm.nr_parallel_seeds != 1:
            raise ValueError("Parallel seeds are not supported")
        self.policy = Policy(config, train_env).to(self.device)
        self.critic = Critic(config, train_env).to(self.device)
        self.networks = {"policy": self.policy, "critic": self.critic}
        self.discriminator = Discriminator(config, train_env).to(self.device)
        self.networks["discriminator"] = self.discriminator
        self.optimizers = {}
        for name, network in self.networks.items():
            lr = config.algorithm.learning_rate
            if name == "discriminator":
                lr = config.algorithm.learning_rate_disc
            elif name == "reward_fn":
                lr = config.algorithm.learning_rate_reward_fn
            elif name == "energyfn":
                lr = config.algorithm.learning_rate_ncsn
            optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4) if name == "energyfn" else torch.optim.Adam(network.parameters(), lr=lr)
            self.optimizers[name] = optimizer
        self.expert_data = {key: torch.as_tensor(value, device=self.device) for key, value in prepare_expert_data(config.algorithm.data_path).items()} if config.runner.mode == "train" else {}
        self.action_low = torch.as_tensor(np.asarray(train_env.single_action_space.low).copy(), dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(np.asarray(train_env.single_action_space.high).copy(), dtype=torch.float32, device=self.device)
        self.H_terminal = (self.action_high - self.action_low).log().sum()
        if config.runner.save_model:
            os.makedirs(self.save_path, exist_ok=True)
        rlx_logger.info(f"Using device: {self.device}")


    def process_action(self, action):
        if self.config.algorithm.action_clipping_and_rescaling:
            action = self.action_low + 0.5 * (action.clamp(-1, 1) + 1) * (self.action_high - self.action_low)
        return action if self.is_torch_data_interface else action.detach().cpu().numpy()


    def log_prob(self, state, action):
        mean, logstd = self.policy(state)
        return (-0.5 * ((action - mean) / logstd.exp()).square() - 0.5 * math.log(2 * math.pi) - logstd).sum(-1)


    def optimize(self, name, loss):
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(self.networks[name].parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.algorithm.max_grad_norm) if name != "energyfn" else torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]))
        cfg = self.config.algorithm
        if cfg.anneal_learning_rate and name in ["policy", "critic", "discriminator"]:
            epochs = cfg.nr_epochs_disc if name == "discriminator" else cfg.nr_epochs
            count = optimizer.state[parameters[0]].get("step", 0)
            count = int(count)
            lr = cfg.learning_rate_disc if name == "discriminator" else cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lr * (1 - (count // (self.nr_minibatches * epochs)) / self.nr_updates)
        optimizer.step()
        return norm.detach()


    def minibatches(self, size, minibatch_size, epochs):
        for epoch in range(epochs):
            indices = torch.randperm(size, device=self.device)[:size // minibatch_size * minibatch_size]
            yield from indices.reshape(-1, minibatch_size)


    def discriminator_loss(self, states, actions, next_states, absorbing, expert_states, expert_actions, expert_next_states, expert_absorbing, log_probs, expert_log_probs):
        cfg = self.config.algorithm
        logits = self.discriminator(states, actions, next_states, absorbing, log_probs)
        expert_logits = self.discriminator(expert_states, expert_actions, expert_next_states, expert_absorbing, expert_log_probs)
        agent_loss = torch.nn.functional.softplus(logits).mean()
        expert_loss = torch.nn.functional.softplus(-expert_logits).mean()
        interpolated = [(cfg.gp_alpha * expert + (1 - cfg.gp_alpha) * agent).detach().requires_grad_(True) for agent, expert in zip([states, actions, next_states], [expert_states, expert_actions, expert_next_states])]
        logits_interpolated = self.discriminator(*interpolated, torch.zeros_like(absorbing), torch.zeros_like(log_probs))
        gradients = torch.autograd.grad(logits_interpolated.sum(), interpolated, create_graph=True, allow_unused=True)
        gradient_norm = torch.sqrt(sum(gradient.square().sum(-1) for gradient in gradients if gradient is not None))
        penalty = (gradient_norm - 1).square().mean()
        loss = agent_loss + expert_loss + cfg.gp_lambda * penalty
        return loss, {"loss/discriminator_loss": loss.detach(), "loss/discriminator_agent_loss": agent_loss.detach(), "loss/discriminator_expert_loss": expert_loss.detach(), "loss/discriminator_gp": penalty.detach()}


    def update(self, batch, learning_iteration_step):
        cfg = self.config.algorithm
        states, next_states, actions, rewards, values, terminations, log_probs, old_means, old_logstds = batch
        shape = rewards.shape
        states = states.flatten(0, 1)
        next_states = next_states.flatten(0, 1)
        actions = actions.flatten(0, 1)
        terminations = terminations.flatten()
        log_probs = log_probs.flatten()
        old_means = old_means.flatten(0, 1)
        old_logstds = old_logstds.reshape(self.batch_size, -1)
        metrics = {}
        expert_indices = torch.randint(len(self.expert_data["states"]), (self.batch_size,), device=self.device)
        expert = {key: value[expert_indices] for key, value in self.expert_data.items()}
        with torch.no_grad():
            expert_log_probs = self.log_prob(expert["states"], expert["actions"])
        disc_metrics = []
        for indices in self.minibatches(self.batch_size, cfg.minibatch_size, cfg.nr_epochs_disc):
            loss, entry = self.discriminator_loss(states[indices], actions[indices], next_states[indices], terminations[indices], expert["states"][indices], expert["actions"][indices], expert["next_states"][indices], expert["absorbing"][indices], log_probs[indices], expert_log_probs[indices])
            self.optimize("discriminator", loss)
            disc_metrics.append(entry)
        metrics.update({key: torch.stack([entry[key] for entry in disc_metrics]).mean() for key in disc_metrics[0]})
        with torch.no_grad():
            logits = self.discriminator(states, actions, next_states, terminations, log_probs)
            absorbing_logits = self.discriminator(next_states, actions, next_states, torch.ones_like(terminations), torch.full_like(log_probs, -self.H_terminal))
            corrected_reward = torch.nn.functional.softplus(logits)
            absorbing_reward = torch.nn.functional.softplus(absorbing_logits)
        with torch.no_grad():
            corrected_reward = cfg.env_reward_frac * rewards + (1 - cfg.env_reward_frac) * corrected_reward.reshape(shape)
            absorbing_reward = (1 - cfg.env_reward_frac) * absorbing_reward
            next_values = self.critic(next_states).reshape(shape)
            terminal = terminations.reshape(shape)
            delta = corrected_reward + cfg.gamma * (1 - terminal) * next_values - values
            if cfg.handle_absorbing_states:
                delta += terminal * cfg.gamma / (1 - cfg.gamma) * (absorbing_reward.reshape(shape) + cfg.entropy_coef * self.H_terminal)
            advantages = torch.empty_like(delta)
            advantages[-1] = delta[-1]
            for step in range(cfg.nr_steps - 2, -1, -1):
                advantages[step] = delta[step] + cfg.gamma * cfg.gae_lambda * (1 - terminal[step]) * advantages[step + 1]
            returns = advantages + values
        all_metrics = []
        all_etas = []
        for indices in self.minibatches(self.batch_size, cfg.minibatch_size, cfg.nr_epochs):
            advantage = advantages.flatten()[indices]
            advantage = (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
            loss, policy_metrics, etas = self.policy_loss(states[indices], actions[indices], log_probs[indices], advantage, old_means[indices], old_logstds[indices], (learning_iteration_step + 1) * self.batch_size)
            policy_metrics["gradients/policy_grad_norm"] = self.optimize("policy", loss)
            critic_loss = 0.5 * (self.critic(states[indices]).squeeze(-1) - returns.flatten()[indices]).square().mean()
            policy_metrics["gradients/critic_grad_norm"] = self.optimize("critic", cfg.critic_coef * critic_loss)
            policy_metrics["loss/critic_loss"] = critic_loss.detach()
            all_metrics.append(policy_metrics)
            all_etas.append(etas.detach())
        metrics.update({key: torch.stack([entry[key].mean() for entry in all_metrics]).mean() for key in all_metrics[0]})
        metrics["lr/learning_rate"] = torch.as_tensor(self.optimizers["policy"].param_groups[0]["lr"], device=self.device)
        metrics["v_value/explained_variance"] = 1 - (returns - values).var(correction=0) / (returns.var(correction=0) + 1e-8)
        metrics["policy/std_dev"] = self.policy.policy_logstd.detach().exp().mean()

        return metrics


    def policy_loss(self, states, actions, old_log_probs, advantages, old_means, old_logstds, global_step):
        cfg = self.config.algorithm
        mean, logstd = self.policy(states)
        etas = torch.zeros(states.shape[0], device=self.device)
        trust_region_loss = torch.zeros((), device=self.device)

        new_log_probs = (-0.5 * ((actions - mean) / logstd.exp()).square() - 0.5 * math.log(2 * math.pi) - logstd).sum(-1)
        entropy = (logstd + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
        logratio = new_log_probs - old_log_probs
        ratio = logratio.exp()
        policy_loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)).mean()
        loss = policy_loss - cfg.entropy_coef * entropy.mean() + 0 * trust_region_loss.mean()
        metrics = {"loss/policy_gradient_loss": policy_loss.detach(), "loss/entropy_loss": entropy.detach().mean(), "policy_ratio/approx_kl": ((ratio - 1) - logratio).detach().mean(), "policy_ratio/clip_fraction": ((ratio - 1).abs() > cfg.clip_range).float().mean()}

        return loss, metrics, etas


    def train(self):

        observation, _ = self.train_env.reset()
        for iteration in range(self.nr_updates):
            started = time.perf_counter()
            transitions = []
            info_values = {}
            for step in range(self.config.algorithm.nr_steps):
                state = torch.as_tensor(observation, device=self.device, dtype=torch.float32).clone()
                with torch.no_grad():
                    mean, logstd = self.policy(state)
                    action = mean + logstd.exp() * torch.randn_like(mean)
                    log_prob = self.log_prob(state, action)
                    value = self.critic(state).squeeze(-1)
                observation, reward, terminated, truncated, info = self.train_env.step(self.process_action(action))
                next_state = torch.as_tensor(observation, device=self.device, dtype=torch.float32).clone()
                done = torch.as_tensor(terminated, device=self.device) | torch.as_tensor(truncated, device=self.device)
                for index in done.nonzero().flatten().tolist():
                    next_state[index] = torch.as_tensor(self.train_env.get_final_observation_at_index(info, index), device=self.device)
                for key, info_value in self.train_env.get_logging_info_dict(info).items():
                    group = "rollout" if key in ["episode_return", "episode_length"] else "env_info"
                    info_values.setdefault(f"{group}/{key}", []).extend(torch.as_tensor(info_value).reshape(-1).tolist())
                transitions.append((state, next_state, action, torch.as_tensor(reward, device=self.device).clone(), value, torch.as_tensor(terminated, device=self.device, dtype=torch.float32).clone(), log_prob, mean, logstd.expand_as(mean).clone()))
            batch = tuple(torch.stack(items).float() for items in zip(*transitions))
            metrics = self.update(batch, iteration)
            global_step = (iteration + 1) * self.batch_size
            metrics.update({key: np.mean(value) for key, value in info_values.items() if value})
            metrics["time/sps"] = self.batch_size / (time.perf_counter() - started)
            metrics["steps/nr_env_steps"] = global_step
            if global_step % self.evaluation_and_save_frequency == 0:
                if self.config.algorithm.evaluation_active:
                    metrics.update(self.evaluate())
                    if self.eval_env is self.train_env:
                        observation, _ = self.train_env.reset()
                if self.config.runner.save_model:
                    self.save()
            self.log(metrics, global_step)


    def evaluate(self, episodes=None):
        observation, _ = self.eval_env.reset()
        returns, lengths = [], []
        step = 0
        while len(returns) < episodes if episodes is not None else step < getattr(self.eval_env, "horizon", 1000):
            with torch.no_grad():
                mean, _ = self.policy(torch.as_tensor(observation, device=self.device, dtype=torch.float32))
            observation, reward, terminated, truncated, info = self.eval_env.step(self.process_action(mean))
            done = torch.as_tensor(terminated) | torch.as_tensor(truncated)
            for index in done.nonzero().flatten().tolist():
                returns.append(float(self.eval_env.get_final_info_value_at_index(info, "episode_return", index)))
                lengths.append(float(self.eval_env.get_final_info_value_at_index(info, "episode_length", index)))
            step += 1
        return {"eval/episode_return": float(np.mean(returns)) if returns else 0.0, "eval/episode_length": float(np.mean(lengths)) if lengths else 0.0}


    def test(self, episodes):
        rlx_logger.info(self.evaluate(episodes))


    def log(self, metrics, step):
        metrics = {key: float(value) for key, value in metrics.items()}
        if self.config.runner.track_console:
            rlx_logger.info(f"Step {step}: {metrics}")
        if self.config.runner.track_tb:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        if self.config.runner.track_wandb:
            wandb.log(metrics, step=step)


    def save(self):
        checkpoint = {"config": self.config.to_dict(), "networks": {key: value.state_dict() for key, value in self.networks.items()}, "optimizers": {key: value.state_dict() for key, value in self.optimizers.items()}}

        torch.save(checkpoint, os.path.join(self.save_path, "latest.model"))


    @staticmethod
    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, map_location="cpu", weights_only=False)
        for key, value in checkpoint["config"]["algorithm"].items():
            if key != "name" and f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = GAIL_PPO(config, train_env, eval_env, run_path, writer)
        for name, network in model.networks.items():
            network.load_state_dict(checkpoint["networks"][name])
            model.optimizers[name].load_state_dict(checkpoint["optimizers"][name])

        return model


    def general_properties():
        return GeneralProperties
