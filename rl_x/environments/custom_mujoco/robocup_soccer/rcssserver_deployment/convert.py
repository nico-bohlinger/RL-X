import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import torch
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy
from rl_x.algorithms.ppo_gru.flax_full_jit.critic import get_critic
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType

from torch_policy import TorchPolicyGRU


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def to_torch_tensor(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32).copy())


def copy_linear(state_dict, torch_prefix, flax_block):
    state_dict[f"{torch_prefix}.weight"] = to_torch_tensor(flax_block["kernel"]).T.contiguous()
    if "bias" in flax_block:
        state_dict[f"{torch_prefix}.bias"] = to_torch_tensor(flax_block["bias"])


def copy_layernorm(state_dict, torch_prefix, flax_block):
    state_dict[f"{torch_prefix}.weight"] = to_torch_tensor(flax_block["scale"])
    state_dict[f"{torch_prefix}.bias"] = to_torch_tensor(flax_block["bias"])


def convert_policy(flax_policy_params, policy_observation_indices):
    p = flax_policy_params

    expected_policy_obs_dim = int(np.asarray(p["gru_obs_encoder_dense"]["kernel"]).shape[0])
    obs_encoding_dim = int(np.asarray(p["gru_obs_encoder_dense"]["kernel"]).shape[1])
    gru_hidden_dim = int(np.asarray(p["gru"]["ir"]["kernel"]).shape[1])
    action_dim = int(np.asarray(p["mean_head"]["kernel"]).shape[1])

    share_gru_obs_encoder = "obs_encoder_dense" not in p
    gru_obs_combine_method = "film" if "gru_film_gamma" in p else "concat"

    if len(policy_observation_indices) != expected_policy_obs_dim:
        raise ValueError(
            f"policy_observation_indices length {len(policy_observation_indices)} does not match "
            f"expected policy obs dim {expected_policy_obs_dim}."
        )

    state_dict = {}

    copy_linear(state_dict, "gru_obs_encoder_dense", p["gru_obs_encoder_dense"])
    copy_layernorm(state_dict, "gru_obs_encoder_ln", p["gru_obs_encoder_ln"])

    if not share_gru_obs_encoder:
        copy_linear(state_dict, "obs_encoder_dense", p["obs_encoder_dense"])
        copy_layernorm(state_dict, "obs_encoder_ln", p["obs_encoder_ln"])

    copy_linear(state_dict, "gru.ir", p["gru"]["ir"])
    copy_linear(state_dict, "gru.iz", p["gru"]["iz"])
    copy_linear(state_dict, "gru.in_proj", p["gru"]["in"])
    copy_linear(state_dict, "gru.hr", p["gru"]["hr"])
    copy_linear(state_dict, "gru.hz", p["gru"]["hz"])
    copy_linear(state_dict, "gru.hn", p["gru"]["hn"])

    copy_layernorm(state_dict, "gru_ln", p["gru_ln"])

    if gru_obs_combine_method == "film":
        copy_linear(state_dict, "gru_film_gamma", p["gru_film_gamma"])
        copy_linear(state_dict, "gru_film_beta", p["gru_film_beta"])

    copy_linear(state_dict, "torso_dense1", p["torso_dense1"])
    copy_layernorm(state_dict, "torso_ln1", p["torso_ln1"])
    copy_linear(state_dict, "torso_dense2", p["torso_dense2"])
    copy_linear(state_dict, "torso_dense3", p["torso_dense3"])
    copy_linear(state_dict, "mean_head", p["mean_head"])

    state_dict["policy_logstd"] = to_torch_tensor(p["policy_logstd"])

    meta = {
        "action_dim": action_dim,
        "obs_encoding_dim": obs_encoding_dim,
        "gru_hidden_dim": gru_hidden_dim,
        "gru_obs_combine_method": gru_obs_combine_method,
        "share_gru_obs_encoder": share_gru_obs_encoder,
        "policy_observation_indices": list(policy_observation_indices),
        "expected_policy_obs_dim": expected_policy_obs_dim,
        "std_dev": float(np.exp(np.asarray(p["policy_logstd"])[0, 0])),
    }

    return state_dict, meta


class SimpleBox:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)


class DummyGeneralProperties:
    action_space_type = ActionSpaceType.CONTINUOUS
    observation_space_type = ObservationSpaceType.FLAT_VALUES


class DummyEnv:
    def __init__(self, obs_dim, action_dim, policy_observation_indices=None):
        self.single_observation_space = SimpleBox(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = SimpleBox(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.general_properties = DummyGeneralProperties()

        if policy_observation_indices is None:
            policy_observation_indices = np.arange(obs_dim, dtype=np.int32)
        self.policy_observation_indices = jnp.array(policy_observation_indices, dtype=jnp.int32)


def build_initial_states(config, env):
    key = jax.random.PRNGKey(config.environment.seed)
    key, policy_key, critic_key = jax.random.split(key, 3)

    policy, _ = get_policy(config, env)
    critic = get_critic(config, env)

    batch_size = config.environment.nr_envs * config.algorithm.nr_steps
    nr_updates = config.algorithm.total_timesteps // batch_size
    nr_minibatches = batch_size // config.algorithm.minibatch_size

    def linear_schedule(count):
        fraction = 1.0 - (count // (nr_minibatches * config.algorithm.nr_epochs)) / nr_updates
        return config.algorithm.learning_rate * fraction

    learning_rate = (
        linear_schedule
        if config.algorithm.anneal_learning_rate
        else config.algorithm.learning_rate
    )

    dummy_obs = jnp.zeros((1, env.single_observation_space.shape[0]), dtype=jnp.float32)
    dummy_policy_gru_carry = policy.initialize_carry(1)

    policy_state = TrainState.create(
        apply_fn=policy.apply,
        params=policy.init(
            policy_key,
            dummy_obs,
            dummy_policy_gru_carry,
            method=policy.apply_one_step,
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(config.algorithm.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
        ),
    )

    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, dummy_obs),
        tx=optax.chain(
            optax.clip_by_global_norm(config.algorithm.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
        ),
    )

    return policy_state, critic_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="locomotion_nn.pth")
    parser.add_argument("--meta-output", type=str, default="locomotion_nn_meta.json")
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--policy-observation-indices", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    meta_output_path = Path(args.meta_output).resolve()

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        shutil.unpack_archive(str(model_path), str(tmpdir), "zip")

        with open(tmpdir / "config_algorithm.json", "r") as f:
            algorithm_cfg_dict = json.load(f)

        if args.policy_observation_indices is None:
            policy_observation_indices = list(range(args.obs_dim))
            logger.warning(
                "No --policy-observation-indices provided. Assuming identity indices [0..%d).",
                args.obs_dim,
            )
        else:
            candidate = Path(args.policy_observation_indices)
            if candidate.exists():
                with candidate.open("r") as f:
                    policy_observation_indices = [int(x) for x in json.load(f)]
            else:
                policy_observation_indices = [
                    int(x.strip()) for x in args.policy_observation_indices.split(",") if x.strip()
                ]

        config = SimpleNamespace(
            runner=SimpleNamespace(
                save_model=False,
                track_console=False,
                track_tb=False,
                track_wandb=False,
            ),
            environment=SimpleNamespace(
                seed=0,
                nr_envs=1,
            ),
            algorithm=to_namespace(algorithm_cfg_dict),
        )

        dummy_env = DummyEnv(
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            policy_observation_indices=policy_observation_indices,
        )

        policy_state, critic_state = build_initial_states(config, dummy_env)

        target = {
            "policy": policy_state,
            "critic": critic_state,
        }

        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = checkpointer.restore(str(tmpdir), item=target, restore_args=restore_args)

        policy_state = restored["policy"]
        flax_policy_params = policy_state.params["params"]

        state_dict, meta = convert_policy(flax_policy_params, policy_observation_indices)

        torch_model = TorchPolicyGRU.from_meta(meta)
        torch_model.load_state_dict(state_dict, strict=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta_output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, output_path)
        with meta_output_path.open("w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved Torch weights to: %s", output_path)
        logger.info("Saved meta JSON to:    %s", meta_output_path)


if __name__ == "__main__":
    main()