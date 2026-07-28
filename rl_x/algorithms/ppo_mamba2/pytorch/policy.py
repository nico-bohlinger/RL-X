import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = Policy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.mamba_obs_combine_method, config.algorithm.share_mamba_obs_encoder, config.algorithm.mamba_d_model, config.algorithm.mamba_num_layers, config.algorithm.mamba_expand, config.algorithm.mamba_state_dim, config.algorithm.mamba_conv_kernel, config.algorithm.mamba_down_projection_dim, config.algorithm.mamba_layer_norm_eps, config.algorithm.mamba_dt_min, config.algorithm.mamba_dt_max, config.algorithm.nr_hidden_units, device, policy_observation_indices).to(device)

    policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
    policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
    return policy


class Mamba2Block(nn.Module):
    def __init__(self, d_model, state_dim, expand, conv_kernel, layer_norm_eps, dt_min, dt_max):
        super().__init__()
        self.inner_dim = d_model * expand
        self.state_dim = state_dim
        self.conv_kernel_size = conv_kernel
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.in_proj = self.layer_init(nn.Linear(d_model, 2 * self.inner_dim))
        self.x_proj = self.layer_init(nn.Linear(self.inner_dim, self.inner_dim + 2 * state_dim))
        self.out_proj = self.layer_init(nn.Linear(self.inner_dim, d_model))
        self.conv_kernel = nn.Parameter(torch.empty(conv_kernel, self.inner_dim))
        nn.init.normal_(self.conv_kernel, std=0.02)
        self.conv_bias = nn.Parameter(torch.zeros(self.inner_dim))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))[None].repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        dt = torch.exp(torch.empty(self.inner_dim).uniform_(np.log(dt_min), np.log(dt_max)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def causal_conv_one_step(self, x, conv_state):
        if self.conv_kernel_size <= 1:
            conv_input = x[:, None]
            next_conv_state = conv_state
        else:
            conv_input = torch.cat([conv_state, x[:, None]], dim=1)
            next_conv_state = conv_input[:, 1:]
        return torch.sum(conv_input * self.conv_kernel[None], dim=1) + self.conv_bias[None], next_conv_state


    def ssm_one_step(self, u, ssm_state):
        params = self.x_proj(u)
        dt_raw = params[..., :self.inner_dim]
        b_t = params[..., self.inner_dim:self.inner_dim + self.state_dim]
        c_t = params[..., self.inner_dim + self.state_dim:]
        dt = torch.nn.functional.softplus(dt_raw + self.dt_bias[None])
        dA = torch.exp(dt[..., None] * -torch.exp(self.A_log)[None])
        next_ssm_state = dA * ssm_state + dt[..., None] * b_t[:, None] * u[..., None]
        return torch.sum(next_ssm_state * c_t[:, None], dim=-1) + self.D[None] * u, next_ssm_state


    def apply_one_step(self, x, carry):
        residual = x
        u, z = torch.chunk(self.in_proj(self.norm(x)), 2, dim=-1)
        u, next_conv_state = self.causal_conv_one_step(u, carry["conv"])
        y, next_ssm_state = self.ssm_one_step(torch.nn.functional.silu(u), carry["ssm"])
        return residual + self.out_proj(y * torch.nn.functional.silu(z)), {"ssm": next_ssm_state, "conv": next_conv_state}


    def forward_sequence(self, x_sequence, done_sequence, carry):
        outputs = []
        for step in range(x_sequence.shape[0]):
            if step > 0:
                reset = 1 - done_sequence[step - 1, :, None]
                carry = {"ssm": carry["ssm"] * reset[..., None], "conv": carry["conv"] * reset[..., None]}
            output, carry = self.apply_one_step(x_sequence[step], carry)
            outputs.append(output)
        return torch.stack(outputs)


class Policy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, mamba_obs_combine_method, share_mamba_obs_encoder, mamba_d_model, mamba_num_layers, mamba_expand, mamba_state_dim, mamba_conv_kernel, mamba_down_projection_dim, mamba_layer_norm_eps, mamba_dt_min, mamba_dt_max, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.mamba_obs_combine_method = mamba_obs_combine_method
        self.share_mamba_obs_encoder = share_mamba_obs_encoder
        self.mamba_d_model = mamba_d_model
        self.mamba_num_layers = mamba_num_layers
        self.mamba_expand = mamba_expand
        self.mamba_state_dim = mamba_state_dim
        self.mamba_conv_kernel = mamba_conv_kernel
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()

        self.mamba_obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), mamba_d_model))
        self.mamba_obs_encoder_ln = nn.LayerNorm(mamba_d_model, eps=mamba_layer_norm_eps)
        if not share_mamba_obs_encoder:
            self.obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), mamba_d_model))
            self.obs_encoder_ln = nn.LayerNorm(mamba_d_model, eps=mamba_layer_norm_eps)
        self.mamba_layers = nn.ModuleList([Mamba2Block(mamba_d_model, mamba_state_dim, mamba_expand, mamba_conv_kernel, mamba_layer_norm_eps, mamba_dt_min, mamba_dt_max) for _ in range(mamba_num_layers)])
        self.mamba_out_ln = nn.LayerNorm(mamba_d_model, eps=mamba_layer_norm_eps)
        self.mamba_down_projection = self.layer_init(nn.Linear(mamba_d_model, mamba_down_projection_dim))
        if mamba_obs_combine_method == "film":
            self.mamba_film_gamma = self.layer_init(nn.Linear(mamba_down_projection_dim, mamba_d_model))
            self.mamba_film_beta = self.layer_init(nn.Linear(mamba_down_projection_dim, mamba_d_model))
        torso_input_dim = mamba_d_model + mamba_down_projection_dim if mamba_obs_combine_method == "concat" else mamba_d_model
        self.torso_dense1 = self.layer_init(nn.Linear(torso_input_dim, nr_hidden_units))
        self.torso_dense2 = self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units))
        self.mean_head = self.layer_init(nn.Linear(nr_hidden_units, action_dim), std=0.01)
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def initialize_carry(self, batch_size):
        inner_dim = self.mamba_d_model * self.mamba_expand
        return {
            "ssm": torch.zeros((batch_size, self.mamba_num_layers, inner_dim, self.mamba_state_dim), dtype=torch.float32, device=self.policy_logstd.device),
            "conv": torch.zeros((batch_size, self.mamba_num_layers, max(self.mamba_conv_kernel - 1, 0), inner_dim), dtype=torch.float32, device=self.policy_logstd.device),
        }


    def mamba_obs_encode(self, obs):
        obs = obs[..., self.policy_observation_indices]
        return torch.nn.functional.elu(self.mamba_obs_encoder_ln(self.mamba_obs_encoder_dense(obs)))


    def obs_encode(self, obs):
        if self.share_mamba_obs_encoder:
            return self.mamba_obs_encode(obs)
        obs = obs[..., self.policy_observation_indices]
        return torch.nn.functional.elu(self.obs_encoder_ln(self.obs_encoder_dense(obs)))


    def decode(self, obs_latent, mamba_latent):
        mamba_latent = torch.nn.functional.elu(self.mamba_out_ln(mamba_latent))
        mamba_latent = torch.nn.functional.elu(self.mamba_down_projection(mamba_latent))
        if self.mamba_obs_combine_method == "concat":
            torso = torch.cat([obs_latent, mamba_latent], dim=-1)
        else:
            torso = obs_latent * self.mamba_film_gamma(mamba_latent) + self.mamba_film_beta(mamba_latent)
        torso = torch.tanh(self.torso_dense1(torso))
        torso = torch.tanh(self.torso_dense2(torso))
        mean = self.mean_head(torso)
        return mean, self.policy_logstd


    def apply_one_step(self, obs, carry):
        x = self.mamba_obs_encode(obs)
        next_ssm = []
        next_conv = []
        for index, layer in enumerate(self.mamba_layers):
            x, next_carry = layer.apply_one_step(x, {"ssm": carry["ssm"][:, index], "conv": carry["conv"][:, index]})
            next_ssm.append(next_carry["ssm"])
            next_conv.append(next_carry["conv"])
        next_carry = {"ssm": torch.stack(next_ssm, dim=1), "conv": torch.stack(next_conv, dim=1)}
        mean, logstd = self.decode(self.obs_encode(obs), x)
        return mean, logstd.expand_as(mean), next_carry


    def forward_sequence(self, obs_sequence, done_sequence, carry):
        x = self.mamba_obs_encode(obs_sequence)
        for index, layer in enumerate(self.mamba_layers):
            x = layer.forward_sequence(x, done_sequence, {"ssm": carry["ssm"][:, index], "conv": carry["conv"][:, index]})
        mean, logstd = self.decode(self.obs_encode(obs_sequence), x)
        return mean, logstd.expand_as(mean)


    def get_action_logprob(self, obs, carry):
        mean, logstd, carry = self.apply_one_step(obs, carry)
        distribution = Normal(mean, logstd.exp())
        action = distribution.sample()
        processed_action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low) if self.action_clipping_and_rescaling else action
        return action, processed_action, distribution.log_prob(action).sum(-1), carry


    def get_logprob_entropy(self, obs, dones, carry, action):
        mean, logstd = self.forward_sequence(obs, dones, carry)
        distribution = Normal(mean, logstd.exp())
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


    def get_deterministic_action(self, obs, carry):
        action, logstd, carry = self.apply_one_step(obs, carry)
        if self.action_clipping_and_rescaling:
            action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        return action, carry
