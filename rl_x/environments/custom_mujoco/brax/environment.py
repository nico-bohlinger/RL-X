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
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.put_model(mj_model)

        self.sys = self.sys.tree_replace({
          'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
          'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
          'opt.iterations': 1,
          'opt.ls_iterations': 4,
        })

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

        observation = self.get_observation(data)
        reward = 0.0
        done = False
        logging_info = {
            "episode_return": reward,
            "episode_length": 0,
            "reward_xy_vel_cmd": 0.0,
            "xy_vel_diff_norm": 0.0,
        }
        info = {
            **logging_info,
            "key": subkey
        }

        return State(data, observation, reward, done, info)


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(self.sys, data.replace(ctrl=action)), None),
            init=state.data,
            xs=(),
            length=self.nr_intermediate_steps
        )

        state.info["episode_length"] += 1

        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data)
        terminated = jnp.logical_or(data.qpos[2] < 0.2, data.qpos[2] > 1.0)
        truncated = state.info["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info["episode_return"] += reward

        def when_done(_):
            __, reset_key = jax.random.split(state.info["key"])
            start_state = self._reset(reset_key)
            start_state = start_state.replace(reward=reward, done=done)
            start_state.info.update(r_info)
            return start_state
        def when_not_done(_):
            return state.replace(data=data, observation=next_observation, reward=reward, done=done)
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
            "reward_xy_vel_cmd": tracking_xy_velocity_command_reward,
            "xy_vel_diff_norm": xy_velocity_difference_norm,
        }

        return reward, info
