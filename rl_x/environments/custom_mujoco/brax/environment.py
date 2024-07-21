from pathlib import Path
from functools import partial
import mujoco
from mujoco import mjx
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.brax.state import State
from rl_x.environments.custom_mujoco.brax.box import Box


class Ant:
    def __init__(self, horizon=1000):
        self.horizon = horizon
        
        xml_path = (Path(__file__).resolve().parent / "data" / "ant.xml").as_posix()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.put_model(mj_model)

        self.nr_intermediate_steps = 5

        initial_height = 0.75
        initial_rotation_quaternion = [0.0, 0.0, 0.0, 1.0]
        initial_joint_angles = [0.0, 0.0] * 4
        self.initial_qpos = jnp.array([0.0, 0.0, initial_height, *initial_rotation_quaternion, *initial_joint_angles])
        self.initial_qvel = jnp.zeros(self.sys.nv)

        self.target_local_x_velocity = 2.0
        self.target_local_y_velocity = 0.0
        
        action_bounds = self.model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.single_action_space = Box(low=action_low, high=action_high, shape=(8,), dtype=jnp.float32)
        self.single_observation_space = Box(low=-jnp.inf, high=jnp.inf, shape=(34,), dtype=jnp.float32)


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        return self._reset(key)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, key):
        key, subkey = jax.random.split(key)

        data = mjx.make_data(self.sys)
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel, ctrl=jnp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)

        next_observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "rollout/episode_return": reward,
            "rollout/episode_length": 0,
            "env_info/reward_xy_vel_cmd": 0.0,
            "env_info/xy_vel_diff_norm": 0.0,
        }
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        return State(data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, subkey)


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)


    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(self.sys, data.replace(ctrl=action)), None),
            init=state.data,
            xs=(),
            length=self.nr_intermediate_steps
        )

        state.info_episode_store["episode_length"] += 1

        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data)
        terminated = jnp.logical_or(data.qpos[2] < 0.2, data.qpos[2] > 1.0)
        truncated = state.info_episode_store["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info_episode_store["episode_return"] += reward
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_length"], state.info["rollout/episode_length"])

        def when_done(_):
            __, reset_key = jax.random.split(state.key)
            start_state = self._reset(reset_key)
            start_state = start_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            start_state.info.update(state.info)  # Keeps only actual last info and discards the reset info
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data):
        global_height = jnp.array([data.qpos[2]])
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]
        local_angular_velocities = data.qvel[3:6]

        inverted_rotation = Rotation.from_quat(data.qpos[3:7]).inv()
        global_linear_velocities = data.qvel[:3]
        local_linear_velocities = inverted_rotation.apply(global_linear_velocities)
        projected_gravity_vector = inverted_rotation.apply(jnp.array([0.0, 0.0, -1.0]))

        observation = jnp.concatenate([
            global_height,
            joint_positions, joint_velocities,
            local_linear_velocities, local_angular_velocities,
            projected_gravity_vector,
            data.ctrl
        ])

        return observation
    
    
    def get_reward(self, data):
        rotation_quaternion = data.qpos[3:7]
        yaw_angle = Rotation.from_quat(rotation_quaternion).as_euler("xyz")[0]
        target_global_x_velocity = self.target_local_x_velocity * jnp.cos(yaw_angle) - self.target_local_y_velocity * jnp.sin(yaw_angle)
        target_global_y_velocity = self.target_local_x_velocity * jnp.sin(yaw_angle) + self.target_local_y_velocity * jnp.cos(yaw_angle)
        target_global_xy_velocity = jnp.array([target_global_x_velocity, target_global_y_velocity])
        current_global_xy_velocity = data.qvel[:2]
        xy_velocity_difference_norm = jnp.sum(jnp.square(target_global_xy_velocity - current_global_xy_velocity))
        tracking_xy_velocity_command_reward = jnp.exp(-xy_velocity_difference_norm / 0.25)

        reward = tracking_xy_velocity_command_reward

        info = {
            "env_info/reward_xy_vel_cmd": tracking_xy_velocity_command_reward,
            "env_info/xy_vel_diff_norm": xy_velocity_difference_norm,
        }

        return reward, info
