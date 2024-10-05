from pathlib import Path
import mujoco
from mujoco import mjx
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.ant_mjx.state import State


class Ant:
    def __init__(self, horizon=100):
        self.horizon = horizon
        
        xml_path = (Path(__file__).resolve().parent / "data" / "ant.xml").as_posix()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.put_model(mj_model)

        self.nr_intermediate_steps = 1

        initial_height = 0.75
        initial_rotation_quaternion = [1.0, 0.0, 0.0, 0.0]  # mujoco quaternion format: [w, x, y, z]
        initial_joint_angles = [0.0, 0.0] * 4
        self.initial_qpos = jnp.array([0.0, 0.0, initial_height, *initial_rotation_quaternion, *initial_joint_angles])
        self.initial_qvel = jnp.zeros(self.sys.nv)

        self.target_local_x_velocity = 2.0
        self.target_local_y_velocity = 0.0


    def reset(self, key):
        key, subkey = jax.random.split(key)

        data = mjx.put_data(self.model, self.data)
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel, ctrl=jnp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)

        observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = False
        logging_info = {
            "episode_return": reward,
            "episode_length": 0,
            "reward_xy_vel_cmd": 0.0,
            "xy_vel_diff_norm": 0.0,
        }
        info = {
            **logging_info,
            "final_observation": jnp.zeros_like(observation),
            "final_info": {**logging_info},
            "done": False,
            "key": subkey
        }

        return State(data, observation, reward, terminated, truncated, info)


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
        terminated = False
        truncated = state.info["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info["episode_return"] += reward
        state.info["done"] = done

        def when_done(_):
            __, reset_key = jax.random.split(state.info["key"])
            start_state = self.reset(reset_key)
            start_state = start_state.replace(reward=reward, terminated=terminated, truncated=truncated)
            start_state.info.update(r_info)
            start_state.info["done"] = True
            start_state.info["final_observation"] = next_observation
            info_keys_to_remove = ["key", "final_observation", "final_info", "done"]
            start_state.info["final_info"] = {key: state.info[key] for key in state.info if key not in info_keys_to_remove}
            return start_state
        def when_not_done(_):
            return state.replace(data=data, observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data):
        global_height = jnp.array([data.qpos[2]])
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]
        local_angular_velocities = data.qvel[3:6]

        base_orientation = [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]  # scipy quaternion format: [x, y, z, w]
        inverted_rotation = Rotation.from_quat(base_orientation).inv()
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
        base_orientation = [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]
        inverted_rotation = Rotation.from_quat(base_orientation).inv()
        current_global_linear_velocity = data.qvel[:3]
        current_local_linear_velocity = inverted_rotation.apply(current_global_linear_velocity)
        target_local_linear_velocity_xy = jnp.array([self.target_local_x_velocity, self.target_local_y_velocity])
        xy_velocity_difference_norm =  jnp.sum(jnp.square(target_local_linear_velocity_xy - current_local_linear_velocity[:2]))
        tracking_xy_velocity_command_reward = jnp.exp(-xy_velocity_difference_norm / 0.25)

        reward = tracking_xy_velocity_command_reward

        info = {
            "reward_xy_vel_cmd": tracking_xy_velocity_command_reward,
            "xy_vel_diff_norm": xy_velocity_difference_norm,
        }

        return reward, info
