from copy import deepcopy
from pathlib import Path

import warp as wp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import io as _mjx_io
from mujoco.mjx._src import types as _mjx_types
import mujoco.mjx.third_party.mujoco_warp._src.io as _mjwp_io
import mujoco.mjx.warp.types as _mjxw_types
from jax.scipy.spatial.transform import Rotation

try:
    from warp._src.jax_experimental.ffi import GraphMode as _WarpGraphMode
    _WARP_GRAPH_MODE_MAP = {
        "jax": _WarpGraphMode.JAX,
        "warp": _WarpGraphMode.WARP,
        "warp_staged": _WarpGraphMode.WARP_STAGED,
        "warp_staged_ex": _WarpGraphMode.WARP_STAGED_EX,
    }
except ImportError:
    _WarpGraphMode = None
    _WARP_GRAPH_MODE_MAP = {}
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.ant.mjx_warp.state import State
from rl_x.environments.custom_mujoco.ant.mjx_warp.box_space import BoxSpace
from rl_x.environments.custom_mujoco.ant.mjx_warp.viewer import MujocoViewer


def _make_batched_warp_data(mj_model, nworld, naconmax, njmax):
    def _wp_to_np(wp_field):
        if isinstance(wp_field, wp.array):
            return wp_field.numpy()
        wp_dtype = type(wp_field)
        if wp_dtype in wp.types.warp_type_to_np_dtype:
            return wp.types.warp_type_to_np_dtype[wp_dtype](wp_field)
        return wp_field

    with wp.ScopedDevice('cpu'):
        dw = _mjwp_io.make_data(mj_model, nworld=nworld, naconmax=naconmax, njmax=njmax)

    fields = _mjx_io._make_data_public_fields(mj_model)
    for k in list(fields.keys()):
        if k in {'userdata', 'plugin_state', 'history'}:
            continue
        if not hasattr(dw, k):
            continue
        fields[k] = _wp_to_np(getattr(dw, k))

    impl_fields = {}
    for k in _mjxw_types.DataWarp.__annotations__.keys():
        raw = _mjx_io._get_nested_attr(dw, k, split='__')
        impl_fields[k] = _wp_to_np(raw)

    eq_active_batched = np.tile(mj_model.eq_active0.reshape(1, -1), (nworld, 1)).astype(bool)

    data = _mjx_types.Data(
        qpos=np.tile(mj_model.qpos0, (nworld, 1)).astype(np.float32),
        eq_active=eq_active_batched,
        **{k: v for k, v in fields.items() if k not in ('qpos', 'eq_active')},
        _impl=_mjxw_types.DataWarp(**impl_fields),
    )
    return jax.device_put(data)


class Ant:
    def __init__(self, env_config):
        self.should_render = env_config.render
        self.horizon = env_config.horizon
        self.action_scaling_factor = env_config.action_scaling_factor
        self.nr_envs = env_config.nr_envs
        self.naconmax = env_config.naconmax
        self.njmax = env_config.njmax
        self.graph_mode = env_config.graph_mode

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "ant.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        gm = _WARP_GRAPH_MODE_MAP.get(self.graph_mode.lower(), None)
        put_model_kwargs = {"impl": "warp"}
        if gm is not None:
            put_model_kwargs["graph_mode"] = gm

        self.mjx_model = mjx.put_model(self.mj_model, **put_model_kwargs)
        self.mjx_data = _make_batched_warp_data(self.mj_model, nworld=self.nr_envs, naconmax=self.naconmax, njmax=self.njmax)

        self.nr_intermediate_steps = 4

        self.initial_qpos = jnp.array(self.mj_model.keyframe("home").qpos)
        self.initial_qvel = jnp.array(self.mj_model.keyframe("home").qvel)

        self.target_local_x_velocity = 2.0
        self.target_local_y_velocity = 0.0

        action_space_size = self.mj_model.nu
        lower_joint_limit, upper_joint_limit = self.mj_model.jnt_range.T
        self.nominal_joint_positions = self.initial_qpos[7:]
        self.single_action_space = BoxSpace(low=lower_joint_limit[1:], high=upper_joint_limit[1:], shape=(action_space_size,), dtype=jnp.float32, center=self.nominal_joint_positions, scale=self.action_scaling_factor)
        self.single_observation_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(34,), dtype=jnp.float32)

        self.viewer = None
        if self.should_render:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)
            c_model = deepcopy(self.mj_model)
            c_data = mujoco.MjData(c_model)
            mujoco.mj_step(c_model, c_data, 1)
            self.light_xdir = c_data.light_xdir
            self.light_xpos = c_data.light_xpos
            self.render_data = mujoco.MjData(self.mj_model)
            del c_model, c_data


    def render(self, state):
        env_id = 0
        self.render_data.qpos[:] = np.array(state.data.qpos[env_id])
        self.render_data.qvel[:] = np.array(state.data.qvel[env_id])
        mujoco.mj_forward(self.mj_model, self.render_data)
        self.render_data.light_xdir = self.light_xdir
        self.render_data.light_xpos = self.light_xpos
        self.viewer.render(self.render_data)


    def reset(self, keys, eval_mode):
        nr_envs = self.nr_envs
        data = self.mjx_data.replace(
            qpos=jnp.tile(self.initial_qpos[None], (nr_envs, 1)),
            qvel=jnp.tile(self.initial_qvel[None], (nr_envs, 1)),
            ctrl=jnp.zeros((nr_envs, self.mj_model.nu)),
        )

        next_observation = self.get_observation(data)
        reward = jnp.zeros(nr_envs, dtype=jnp.float32)
        terminated = jnp.zeros(nr_envs, dtype=bool)
        truncated = jnp.zeros(nr_envs, dtype=bool)
        info = {
            "rollout/episode_return": jnp.zeros(nr_envs, dtype=jnp.float32),
            "rollout/episode_length": jnp.zeros(nr_envs, dtype=jnp.float32),
            "env_info/reward_xy_vel_cmd": jnp.zeros(nr_envs, dtype=jnp.float32),
            "env_info/xy_vel_diff_norm": jnp.zeros(nr_envs, dtype=jnp.float32),
        }
        info_episode_store = {
            "episode_return": jnp.zeros(nr_envs, dtype=jnp.float32),
            "episode_length": jnp.zeros(nr_envs, dtype=jnp.float32),
        }

        return State(data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, keys[:nr_envs])


    def step(self, state, action):
        ctrl = self.nominal_joint_positions + action * self.action_scaling_factor
        data = jax.lax.fori_loop(0, self.nr_intermediate_steps, lambda _, d: mjx.step(self.mjx_model, d), state.data.replace(ctrl=ctrl))

        obs = self.get_observation(data)
        reward, r_info = self.get_reward(data)

        height = data.qpos[:, 2]
        terminated = (height < 0.2) | (height > 1.0)
        episode_length = state.info_episode_store["episode_length"] + 1.0
        truncated = episode_length >= self.horizon
        done = terminated | truncated

        episode_return = state.info_episode_store["episode_return"] + reward
        info = {
            "rollout/episode_return": jnp.where(done, episode_return, state.info["rollout/episode_return"]),
            "rollout/episode_length": jnp.where(done, episode_length, state.info["rollout/episode_length"]),
            "env_info/reward_xy_vel_cmd": r_info["env_info/reward_xy_vel_cmd"],
            "env_info/xy_vel_diff_norm": r_info["env_info/xy_vel_diff_norm"],
        }
        info_episode_store = {
            "episode_return": jnp.where(done, jnp.zeros_like(episode_return), episode_return),
            "episode_length": jnp.where(done, jnp.zeros_like(episode_length), episode_length),
        }

        done_col = done[:, None]
        final_data = data.replace(
            qpos=jnp.where(done_col, jnp.tile(self.initial_qpos[None], (self.nr_envs, 1)), data.qpos),
            qvel=jnp.where(done_col, jnp.tile(self.initial_qvel[None], (self.nr_envs, 1)), data.qvel),
            ctrl=jnp.where(done_col, jnp.zeros_like(data.ctrl), data.ctrl),
        )

        return State(final_data, self.get_observation(final_data), obs, reward, terminated, truncated, info, info_episode_store, state.key)


    def get_observation(self, data):
        global_height = data.qpos[:, 2:3]
        joint_positions = data.qpos[:, 7:] - self.nominal_joint_positions
        joint_velocities = data.qvel[:, 6:]
        local_angular_velocities = data.qvel[:, 3:6]

        base_orientation = jnp.stack([data.qpos[:, 4], data.qpos[:, 5], data.qpos[:, 6], data.qpos[:, 3]], axis=-1)
        inverted_rotation = Rotation.from_quat(base_orientation).inv()
        local_linear_velocities = inverted_rotation.apply(data.qvel[:, :3])
        projected_gravity_vector = inverted_rotation.apply(jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (self.nr_envs, 3)))

        observation = jnp.concatenate([global_height, joint_positions, joint_velocities, local_linear_velocities, local_angular_velocities, projected_gravity_vector, data.ctrl], axis=-1)
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -100.0, 100.0)
        return observation


    def get_reward(self, data):
        base_orientation = jnp.stack([data.qpos[:, 4], data.qpos[:, 5], data.qpos[:, 6], data.qpos[:, 3]], axis=-1)
        inverted_rotation = Rotation.from_quat(base_orientation).inv()
        current_local_linear_velocity = inverted_rotation.apply(data.qvel[:, :3])
        target_local_linear_velocity_xy = jnp.array([self.target_local_x_velocity, self.target_local_y_velocity])
        xy_velocity_difference_norm = jnp.sum(jnp.square(target_local_linear_velocity_xy - current_local_linear_velocity[:, :2]), axis=-1)
        tracking_xy_velocity_command_reward = jnp.exp(-xy_velocity_difference_norm / 0.25)

        reward = tracking_xy_velocity_command_reward
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = jnp.clip(reward, -10.0, 10.0)

        info = {
            "env_info/reward_xy_vel_cmd": tracking_xy_velocity_command_reward,
            "env_info/xy_vel_diff_norm": xy_velocity_difference_norm,
        }
        return reward, info


    def close(self):
        if self.viewer:
            self.viewer.close()
