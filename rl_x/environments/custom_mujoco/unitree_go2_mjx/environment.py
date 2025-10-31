from copy import deepcopy
from pathlib import Path
from functools import partial
import mujoco
from mujoco import mjx
from dm_control import mjcf
import pygame
import numpy as np
from scipy.spatial.transform import Rotation as Rotation_NP
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.unitree_go2_mjx.state import State
from rl_x.environments.custom_mujoco.unitree_go2_mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.unitree_go2_mjx.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.unitree_go2_mjx.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.action_delay_functions.handler import get_get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.control_functions.handler import get_domain_randomization_control_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.observation_dropout_functions.handler import get_observation_dropout_function
from rl_x.environments.custom_mujoco.unitree_go2_mjx.terrain_functions.handler import get_terrain_function


class GeneralLocomotionEnv:
    def __init__(self,
                 robot_config, runner_mode, render,
                 control_type, command_type, command_sampling_type, initial_state_type,
                 reward_type, termination_type,
                 domain_randomization_sampling_type, domain_randomization_action_delay_type,
                 domain_randomization_mujoco_model_type, domain_randomization_control_type,
                 domain_randomization_perturbation_type, domain_randomization_perturbation_sampling_type,
                 observation_noise_type, observation_dropout_type,
                 terrain_type,
                 add_goal_arrow, timestep, episode_length_in_seconds, total_nr_envs):
        
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.add_goal_arrow = add_goal_arrow
        self.total_nr_envs = total_nr_envs

        self.terrain_function = get_terrain_function(terrain_type, self)
        xml_path = (self.robot_config["directory_path"] / "data" / self.terrain_function.xml_file_name).as_posix()
        if self.should_render and self.add_goal_arrow:
            xml_handle = mjcf.from_path(xml_path)
            trunk = xml_handle.find("body", "trunk")
            trunk.add("body", name="dir_arrow", pos="0 0 0.15")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
            self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        else:
            self.initial_mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.initial_mj_model.opt.timestep = timestep
        self.data = mujoco.MjData(self.initial_mj_model)
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        c_model = deepcopy(self.initial_mj_model)
        c_data = mujoco.MjData(c_model)
        mujoco.mj_step(c_model, c_data, 1)
        
        self.joint_nominal_positions = jnp.array(self.initial_mj_model.keyframe("home").qpos[7:])
        self.joint_max_velocities = jnp.array(robot_config["joint_max_velocities"])
        self.max_qvel = jnp.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, *self.joint_max_velocities])
        self.motor_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        obs_idx.update_nr_joints(len(self.motor_joint_names))
        self.joint_observation_indices = [joint_indices for joint_indices in obs_idx.JOINTS.reshape(len(self.motor_joint_names), -1)]
        
        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = jnp.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        obs_idx.update_nr_feet(len(self.feet_names))
        self.feet_observation_indices = [foot_indices for foot_indices in obs_idx.FEET.reshape(len(self.feet_names), -1)]

        feet_xpos = c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, jnp.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = jnp.array([list(pair) for pair in feet_symmetry_set])

        self.control_function = get_control_function(control_type, self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / timestep))
        self.dt = timestep * self.nr_substeps
        self.horizon = int(round(episode_length_in_seconds * self.control_frequency_hz))
        self.command_function = get_command_function(command_type, self)
        self.command_sampling_function = get_sampling_function(command_sampling_type, self)
        self.initial_state_function = get_initial_state_function(initial_state_type, self)
        self.reward_function = get_reward_function(reward_type, self)
        self.termination_function = get_termination_function(termination_type, self)
        self.domain_randomization_sampling_function = get_sampling_function(domain_randomization_sampling_type, self)
        self.domain_randomization_action_delay_function = get_get_domain_randomization_action_delay_function(domain_randomization_action_delay_type, self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(domain_randomization_mujoco_model_type, self)
        self.domain_randomization_control_function = get_domain_randomization_control_function(domain_randomization_control_type, self)
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(domain_randomization_perturbation_type, self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(domain_randomization_perturbation_sampling_type, self)
        self.observation_noise_function = get_observation_noise_function(observation_noise_type, self)
        self.observation_dropout_function = get_observation_dropout_function(observation_dropout_type, self)

        self.terrain_function.init_attributes_with_model()
        
        self.single_action_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(self.initial_mjx_model.nu,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(obs_idx.OBSERVATION_SIZE,), dtype=jnp.float32)

        self.viewer = None
        if self.should_render:
            self.viewer = MujocoViewer(c_model, self.dt)
            self.light_xdir = c_data.light_xdir
            self.light_xpos = c_data.light_xpos

            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True
        else:
            del c_model
        del c_data

    
    def render(self, state):
        mjx_model = state.mjx_model
        mj_model = self.viewer.model
        for field in mjx.Model.fields():
            if field.type in [jax.Array, np.ndarray]:
                field_name = field.name
                if field.name in ["mesh_conver", "dof_hasfrictionloss", "tendon_hasfrictionloss", "_sizes"]:
                    continue
                if field_name in "geom_rbound_hfield":
                    field_name = "geom_rbound"
                mjx_value = getattr(mjx_model, field_name)
                mj_value = getattr(mj_model, field_name)
                if mjx_value.shape != mj_value.shape:
                    mjx_value = mjx_value.reshape(mj_value.shape)
                setattr(mj_model, field_name, mjx_value)
        if self.terrain_function.uses_hfield and state.info_episode_store["episode_step"] == 1:
            mujoco.mjr_uploadHField(mj_model, self.viewer.context, 0)

        env_id = 0
        data = mjx.get_data(mj_model, state.data)[env_id]

        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos

        if self.joystick_present:
            pygame.event.pump()
            goal_x_velocity = -self.joystick.get_axis(1)
            goal_y_velocity = -self.joystick.get_axis(0)
            goal_yaw_velocity = -self.joystick.get_axis(3)
            explicit_commands = True
        elif Path("commands.txt").is_file():
            with open("commands.txt", "r") as f:
                commands = f.readlines()
            if len(commands) == 3:
                goal_x_velocity = float(commands[0])
                goal_y_velocity = float(commands[1])
                goal_yaw_velocity = float(commands[2])
                explicit_commands = True

        if explicit_commands and self.runner_mode == "test":
            state.internal_state["goal_velocities"] = jnp.tile(jnp.array([goal_x_velocity, goal_y_velocity, goal_yaw_velocity]), (self.total_nr_envs, 1)) 

        if self.add_goal_arrow:
            goal_velocities = state.internal_state["goal_velocities"][env_id]
            trunk_rotation = state.internal_state["orientation_euler"][env_id][2]
            desired_angle = trunk_rotation + np.arctan2(goal_velocities[1], goal_velocities[0])
            rot_mat = Rotation_NP.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([goal_velocities[0], goal_velocities[1]])))
            mj_model.site_size[1, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            data.site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            data.site("dir_arrow_ball").xpos = data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]
        
        self.viewer.render(data)

        return state
    

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        mjx_model = self.initial_mjx_model
        data = mjx.make_data(mjx_model)

        next_observation = jnp.zeros(obs_idx.OBSERVATION_SIZE, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False

        internal_state = {
            "total_timesteps": 0,
            "reached_max_timesteps": False,
            "goal_velocities": jnp.array([0.0, 0.0, 0.0]),
            "orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "orientation_euler": jnp.array([0.0, 0.0, 0.0]),
            "last_action": jnp.zeros(self.initial_mjx_model.nu),
        }
        self.command_function.init(internal_state)
        self.control_function.init(internal_state)
        self.reward_function.init(internal_state, mjx_model)
        self.terrain_function.init(internal_state)

        info = {}
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros(self.initial_mjx_model.nu), info)
        info["rollout/episode_return"] = reward
        info["rollout/episode_length"] = 0
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
        }

        state = State(mjx_model, data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, internal_state, key)
        
        return self._reset(state)


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _vmap_reset(self, state):
        return self._reset(state)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, initial_state_key, terrain_key, domain_randomization_key, observation_key = jax.random.split(state.key, 5)
        state = state.replace(key=key)

        mjx_model = self.terrain_function.sample(state.mjx_model, state.internal_state, state.info, terrain_key)

        data = mjx.make_data(mjx_model)
        qpos, qvel = self.initial_state_function.setup(data, mjx_model, state.internal_state, initial_state_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.initial_mjx_model.nu))
        # data = mjx.forward(self.initial_mjx_model, data)

        new_state = state
        new_state.internal_state["orientation_rotation"] = Rotation.from_quat([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])
        new_state.internal_state["orientation_rotation_inverse"] = new_state.internal_state["orientation_rotation"].inv()
        new_state.internal_state["orientation_euler"] = new_state.internal_state["orientation_rotation"].as_euler("xyz")
        new_state.internal_state["last_action"] = jnp.zeros(self.initial_mjx_model.nu)
        self.reward_function.setup(new_state.internal_state)
        self.termination_function.setup(new_state.internal_state)
        self.domain_randomization_action_delay_function.setup(new_state.internal_state)
        data, mjx_model = self.handle_domain_randomization(new_state.internal_state, mjx_model, data, domain_randomization_key, is_setup=True)

        next_observation = self.get_observation(data, mjx_model, new_state.internal_state, observation_key, jnp.zeros(self.initial_mjx_model.nu))
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
        }

        # Reset everything besides parts of the internal_state, info and the key
        new_state = new_state.replace(
            mjx_model=mjx_model,
            data=data,
            next_observation=next_observation, actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated, truncated=truncated,
            info_episode_store=info_episode_store
        )

        return new_state


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)


    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, action_delay_key, domain_randomization_key, command_sampling_key, command_key, observation_key, terrain_key = jax.random.split(state.key, 7)
        state = state.replace(key=key)

        action = self.domain_randomization_action_delay_function.delay_action(action, state.internal_state, action_delay_key)

        torques = self.control_function.process_action(action, state.internal_state, state.data)

        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(state.mjx_model, data.replace(ctrl=torques)), None),
            init=state.data,
            xs=(),
            length=self.nr_substeps
        )
        data = data.replace(qvel=jnp.clip(data.qvel, -self.max_qvel, self.max_qvel))

        state.internal_state["orientation_rotation"] = Rotation.from_quat([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])
        state.internal_state["orientation_rotation_inverse"] = state.internal_state["orientation_rotation"].inv()
        state.internal_state["orientation_euler"] = state.internal_state["orientation_rotation"].as_euler("xyz")

        data, mjx_model = self.handle_domain_randomization(state.internal_state, state.mjx_model, data, domain_randomization_key)
        state = state.replace(data=data, mjx_model=mjx_model)

        self.terrain_function.pre_step(data, state.internal_state)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, action, state.info)

        should_sample_commands = self.command_sampling_function.step(command_sampling_key)
        self.command_function.get_next_command(state.data, state.internal_state, should_sample_commands, command_key)
        
        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, action)
        terminated = self.termination_function.should_terminate(data, state.internal_state, torques)
        truncated = state.info_episode_store["episode_step"] >= self.horizon
        done = terminated | truncated

        data = self.terrain_function.post_step(data, mjx_model, state.internal_state, terrain_key)
        self.reward_function.step(data, mjx_model, state.internal_state)

        state.internal_state["total_timesteps"] += self.total_nr_envs
        state.internal_state["reached_max_timesteps"] = (state.internal_state["total_timesteps"] >= 2e9) | state.internal_state["reached_max_timesteps"]
        state.internal_state["last_action"] = action
        state.info_episode_store["episode_step"] += 1
        state.info_episode_store["episode_return"] += reward
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_step"], state.info["rollout/episode_length"])

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data, mjx_model, internal_state, key, action):
        observation = jnp.zeros(obs_idx.OBSERVATION_SIZE, dtype=jnp.float32)

        # Dynamic observations
        for i, joint_range in enumerate(self.joint_observation_indices):
            observation = observation.at[joint_range[0]].set(data.qpos[i+7] - self.joint_nominal_positions[i])
            observation = observation.at[joint_range[1]].set(data.qvel[i+6])
            observation = observation.at[joint_range[2]].set(action[i])

        for i, foot_range in enumerate(self.feet_observation_indices):
            foot_name = self.feet_names[i]
            observation = observation.at[foot_range[0]].set(self.terrain_function.check_foot_floor_contact(data, mjx_model, internal_state, foot_name))
            touchdown_name = f"time_since_last_touchdown_{foot_name.replace('_foot', '').lower()}"
            observation = observation.at[foot_range[1]].set(internal_state[touchdown_name])

        # General observations
        trunk_linear_velocity = internal_state["orientation_rotation_inverse"].apply(data.qvel[:3])
        observation = observation.at[obs_idx.TRUNK_LINEAR_VELOCITIES].set(trunk_linear_velocity)

        trunk_angular_velocity = data.qvel[3:6]
        observation = observation.at[obs_idx.TRUNK_ANGULAR_VELOCITIES].set(trunk_angular_velocity)

        goal_velocities = internal_state["goal_velocities"]
        observation = observation.at[obs_idx.GOAL_VELOCITIES].set(goal_velocities)

        projected_gravity_vector = internal_state["orientation_rotation_inverse"].apply(jnp.array([0.0, 0.0, -1.0]))
        observation = observation.at[obs_idx.PROJECTED_GRAVITY].set(projected_gravity_vector)

        observation = observation.at[obs_idx.HEIGHT].set(internal_state["robot_height_over_ground"])

        # Add noise
        observation = self.observation_noise_function.modify_observation(observation, key)

        # Dropout
        observation = self.observation_dropout_function.modify_observation(observation, key)

        # Normalize and clip
        for i, joint_range in enumerate(self.joint_observation_indices):
            observation = observation.at[joint_range[0]].set(observation[joint_range[0]] / 3.14)
            observation = observation.at[joint_range[1]].set(observation[joint_range[1]] / self.joint_max_velocities[i])
            observation = observation.at[joint_range[2]].set(observation[joint_range[2]] / 3.14)
        for i, foot_range in enumerate(self.feet_observation_indices):
            observation = observation.at[foot_range[1]].set(jnp.minimum(jnp.maximum(observation[foot_range[1]], 0.0), 5.0))
        observation = observation.at[obs_idx.TRUNK_ANGULAR_VELOCITIES].set(observation[obs_idx.TRUNK_ANGULAR_VELOCITIES] / 10.0)
        observation = observation.at[obs_idx.TRUNK_LINEAR_VELOCITIES].set(observation[obs_idx.TRUNK_LINEAR_VELOCITIES] / 10.0)

        return observation
    

    def handle_domain_randomization(self, internal_state, mjx_model, data, key, is_setup=False):
        domain_sampling_key, domain_perturbation_sampling_key, control_key, mujoco_model_key, action_delay_key, perturbation_key = jax.random.split(key, 6)

        if is_setup:
            should_randomize_domain = self.domain_randomization_sampling_function.setup(domain_sampling_key)
            should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.setup(domain_perturbation_sampling_key)
        else:
            should_randomize_domain = self.domain_randomization_sampling_function.step(domain_sampling_key)
            should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.step(domain_perturbation_sampling_key)
        
        self.domain_randomization_control_function.sample(internal_state, should_randomize_domain, control_key)
        mjx_model = self.domain_randomization_mujoco_model_function.sample(mjx_model, should_randomize_domain, mujoco_model_key)
        self.domain_randomization_action_delay_function.sample(internal_state, should_randomize_domain, action_delay_key)
        self.reward_function.handle_model_change(internal_state, mjx_model, should_randomize_domain)
        
        data = self.domain_randomization_perturbation_function.sample(data, should_randomize_domain_perturbation, perturbation_key)

        return data, mjx_model
    

    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
