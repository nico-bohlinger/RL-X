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

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.state import State
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.handler import get_terrain_function


class LocomotionEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.nr_envs = nr_envs

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)

        # Remove all unnecessary assets, materials, meshes and geoms during training
        # This removes all geoms besides feet and floor, if the contacts for other geoms should be enabled this needs to be changed
        # Also if you want to render the training, the lines can be commented out
        for texture in xml_handle.asset.find_all("texture"):
            texture.remove()
        for material in xml_handle.asset.find_all("material"):
            material.remove()
        for mesh in xml_handle.asset.find_all("mesh"):
            mesh.remove()
        for geom in xml_handle.find_all("geom"):
            is_foot_geom = geom.name and "foot" in geom.name
            is_floor_geom = geom.name == "floor"
            is_reward_collision_sphere_geom = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
            if not is_foot_geom and not is_floor_geom and not is_reward_collision_sphere_geom:
                geom.remove()
            if is_floor_geom:
                geom.material = ""

        if "hfield" in env_config["terrain"]["type"]:
            xml_handle.asset.insert("hfield", 0, name="empty_hfield", file="default_hfield_80.png", size="4 4 30.0 0.125")
            floor = xml_handle.find("geom", "floor")
            floor.type = "hfield"
            floor.hfield = "empty_hfield"
        
        if self.should_render and self.add_goal_arrow:
            trunk = xml_handle.find("body", "trunk")
            trunk.add("body", name="dir_arrow", pos="0 0 0.15")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
        
        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.data = mujoco.MjData(self.initial_mj_model)
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.make_data(self.initial_mjx_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)  # Necessary because of error with toddlerbot
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)
        
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.actuator_joint_max_velocities = jnp.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = jnp.array(self.initial_mj_model.keyframe("home").qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = jnp.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = jnp.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = jnp.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nr_joints = self.initial_mj_model.njnt

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor("imu_angular_velocity").id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor("imu_linear_velocity").id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = jnp.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, jnp.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = jnp.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = jnp.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_box = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = jnp.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])

        body_to_parentid = jnp.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = jnp.array([jnp.sum(body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = jnp.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.reward_collision_sphere_geom_ids = jnp.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])

        self.has_equality_constraints = len(self.initial_mj_model.eq_data) > 0

        self.robot_dimensions_mean = 0.5  # This can be calculated smartly...

        self.env_curriculum_nr_levels = env_config["env_curriculum_nr_levels"]
        self.env_curriculum_level_success_episode_return = env_config["env_curriculum_level_success_episode_return"]

        self.control_function = get_control_function(env_config["control_type"], self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        self.command_function = get_command_function(env_config["command"]["type"], self)
        self.command_sampling_function = get_sampling_function(env_config["command"]["sampling_type"], self)
        self.initial_state_function = get_initial_state_function(env_config["domain_randomization"]["initial_state"]["type"], self)
        self.reward_function = get_reward_function(env_config["reward"]["type"], self)
        self.termination_function = get_termination_function(env_config["termination"]["type"], self)
        self.policy_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["policy_exteroceptive_observation_type"], self)
        self.critic_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["critic_exteroceptive_observation_type"], self)
        self.terrain_function = get_terrain_function(env_config["terrain"]["type"], self)
        self.domain_randomization_sampling_function = get_sampling_function(env_config["domain_randomization"]["sampling_type"], self)
        self.domain_randomization_action_delay_function = get_domain_randomization_action_delay_function(env_config["domain_randomization"]["action_delay"]["type"], self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(env_config["domain_randomization"]["mujoco_model"]["type"], self)
        self.domain_randomization_seen_robot_function = get_domain_randomization_seen_robot_function(env_config["domain_randomization"]["seen_robot"]["type"], self)
        self.domain_randomization_unseen_robot_function = get_domain_randomization_unseen_robot_function(env_config["domain_randomization"]["unseen_robot"]["type"], self)
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(env_config["domain_randomization"]["perturbation"]["type"], self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(env_config["domain_randomization"]["perturbation"]["sampling_type"], self)
        self.observation_noise_function = get_observation_noise_function(env_config["domain_randomization"]["observation_noise"]["type"], self)
        self.joint_dropout_function = get_joint_dropout_function(env_config["domain_randomization"]["joint_dropout"]["type"], self)
        
        action_space_size = self.nr_actuator_joints
        self.single_action_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(action_space_size,), dtype=jnp.float32)

        self.single_observation_space = self.get_observation_space()

        self.observation_noise_function.init_attributes()

        if self.should_render:
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)

            self.dir_arrow_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "dir_arrow")
            self.uses_hfield = self.initial_mj_model.hfield_data.shape[0] != 0
            self.light_xdir = self.c_data.light_xdir
            self.light_xpos = self.c_data.light_xpos

            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True
        del self.c_model, self.c_data

    
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
        if self.uses_hfield and state.info_episode_store["episode_step"] == 1:
            mujoco.mjr_uploadHField(mj_model, self.viewer.context, 0)

        env_id = 0
        data = mjx.get_data(mj_model, state.data)[env_id]

        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos

        if self.runner_mode == "test":
            explicit_velocity_commands = False
            if self.joystick_present:
                pygame.event.pump()
                goal_x_velocity = -self.joystick.get_axis(1)
                goal_y_velocity = -self.joystick.get_axis(0)
                goal_yaw_velocity = -self.joystick.get_axis(3)
                explicit_velocity_commands = True
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == 3:
                    goal_x_velocity = float(commands[0])
                    goal_y_velocity = float(commands[1])
                    goal_yaw_velocity = float(commands[2])
                    explicit_velocity_commands = True
            if explicit_velocity_commands:
                goal_velocities = jnp.array([goal_x_velocity, goal_y_velocity, goal_yaw_velocity])
                goal_velocities = jnp.where(jnp.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * state.internal_state["max_command_velocity"]), 0.0, goal_velocities)
                goal_velocities = jnp.clip(goal_velocities, -state.internal_state["max_command_velocity"], state.internal_state["max_command_velocity"])
                state.internal_state["goal_velocities"] = jnp.tile(goal_velocities, (self.nr_envs, 1))
                actuator_joint_keep_nominal = jnp.where(jnp.all(goal_velocities == 0.0), jnp.ones(self.nr_actuator_joints, dtype=bool), self.command_function.default_actuator_joint_keep_nominal)
                state.internal_state["actuator_joint_keep_nominal"] = jnp.tile(actuator_joint_keep_nominal, (self.nr_envs, 1))
            explicit_actuator_keep_nominal_commands = False
            if Path(f"commands.txt").is_file():
                with open(f"commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == self.nr_actuator_joints:
                    actuator_keep_nominal_commands = [bool(int(command.replace("\n",""))) for command in commands]
                    actuator_keep_nominal_commands = jnp.array(actuator_keep_nominal_commands)
                    explicit_actuator_keep_nominal_commands = True
            if explicit_actuator_keep_nominal_commands:
                state.internal_state["actuator_joint_keep_nominal"] = jnp.tile(actuator_keep_nominal_commands, (self.nr_envs, 1))

        if self.add_goal_arrow:
            goal_velocities = state.internal_state["goal_velocities"][env_id]
            trunk_rotation = state.internal_state["imu_orientation_euler"][env_id][2]
            desired_angle = trunk_rotation + np.arctan2(goal_velocities[1], goal_velocities[0])
            rot_mat = Rotation_NP.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([goal_velocities[0], goal_velocities[1]])))
            mj_model.site_size[self.dir_arrow_id, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            data.site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            data.site("dir_arrow_ball").xpos = data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]
        
        self.viewer.render(data)

        return state
    

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        mjx_model = self.initial_mjx_model
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False


        internal_state = {
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": jnp.where(eval_mode, 1.0, 0.0),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": jnp.array([0.0, 0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": jnp.array([0.0, 0.0, 0.0]),
            "last_action": jnp.zeros(self.nr_actuator_joints),
            "second_last_action": jnp.zeros(self.nr_actuator_joints),
            "joint_dropout_mask": jnp.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "max_command_velocity": jnp.minimum(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity),
            "nr_collisions_in_nominal": 0,
        }
        self.command_function.init(internal_state)
        self.reward_function.init(internal_state, mjx_model)
        self.terrain_function.init(internal_state)
        self.joint_dropout_function.init(internal_state)
        self.domain_randomization_action_delay_function.init(internal_state)
        self.domain_randomization_seen_robot_function.init(internal_state)
        self.domain_randomization_unseen_robot_function.init(internal_state)

        info = {}
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros(self.nr_actuator_joints), info)
        info["rollout/episode_return"] = reward
        info["rollout/episode_length"] = 0
        info["env_curriculum/coefficient"] = internal_state["env_curriculum_coeff"]
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
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

        mjx_model = self.terrain_function.sample(state.mjx_model, state.internal_state, terrain_key)

        data = self.mjx_data
        qpos, qvel = self.initial_state_function.setup(mjx_model, state.internal_state, initial_state_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.nr_actuator_joints))
        # data = mjx.forward(self.initial_mjx_model, data)

        new_state = state

        episode_success = new_state.info_episode_store["episode_return"] >= self.env_curriculum_level_success_episode_return
        new_state.internal_state["env_curriculum_levels_in_a_row"] = jnp.where(episode_success,
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] + 1,
                1
            ),
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] < 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] - 1,
                -1
            )
        )
        new_state.internal_state["env_curriculum_coeff"] =  jnp.clip(new_state.internal_state["env_curriculum_coeff"] + new_state.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)
        new_state.internal_state["env_curriculum_coeff"] = jnp.where(new_state.internal_state["in_eval_mode"], 1.0, new_state.internal_state["env_curriculum_coeff"])

        new_state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        new_state.internal_state["imu_orientation_rotation_inverse"] = new_state.internal_state["imu_orientation_rotation"].inv()
        new_state.internal_state["imu_orientation_euler"] = new_state.internal_state["imu_orientation_rotation"].as_euler("xyz")
        new_state.internal_state["last_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["second_last_action"] = jnp.zeros(self.nr_actuator_joints)
        self.reward_function.setup(new_state.internal_state)
        self.domain_randomization_action_delay_function.setup(new_state.internal_state)
        data, mjx_model = self.handle_domain_randomization(new_state.internal_state, mjx_model, data, domain_randomization_key, is_episode_start=True)

        next_observation = self.get_observation(data, mjx_model, new_state.internal_state, observation_key, jnp.zeros(self.nr_actuator_joints))
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
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

        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action, state.internal_state, action_delay_key)

        target_joint_positions = self.control_function.process_action(delayed_action, state.internal_state)

        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(state.mjx_model, data.replace(ctrl=target_joint_positions)), None),
            init=state.data,
            xs=(),
            length=self.nr_substeps
        )
        max_qvel = 100 * jnp.ones(self.initial_mj_model.nv)
        max_qvel = max_qvel.at[self.actuator_joint_mask_qvel].set(state.internal_state["actuator_joint_max_velocities"])
        data = data.replace(qvel=jnp.clip(data.qvel, -max_qvel, max_qvel))

        state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        state.internal_state["imu_orientation_rotation_inverse"] = state.internal_state["imu_orientation_rotation"].inv()
        state.internal_state["imu_orientation_euler"] = state.internal_state["imu_orientation_rotation"].as_euler("xyz")

        data, mjx_model = self.handle_domain_randomization(state.internal_state, state.mjx_model, data, domain_randomization_key)
        state = state.replace(data=data, mjx_model=mjx_model)

        self.terrain_function.pre_step(data, state.internal_state)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)

        should_sample_commands = self.command_sampling_function.step(command_sampling_key)
        self.command_function.get_next_command(state.internal_state, should_sample_commands, command_key)
        
        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        terminated = self.termination_function.should_terminate(state.internal_state) | jnp.any(jnp.abs(data.qvel[:3]) == 100.0)
        truncated = state.info_episode_store["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        data = self.terrain_function.post_step(data, mjx_model, state.internal_state, terrain_key)
        self.reward_function.step(data, state.internal_state)

        state.internal_state["second_last_action"] = state.internal_state["last_action"]
        state.internal_state["last_action"] = chosen_action
        state.info_episode_store["episode_step"] += 1
        state.info_episode_store["episode_return"] += reward
        state.info_episode_store["episode_total_xy_velocity_diff_abs"] += state.info["env_info/xy_vel_diff_abs"]
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_step"], state.info["rollout/episode_length"])
        state.info["env_curriculum/coefficient"] = state.internal_state["env_curriculum_coeff"]

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data, mjx_model, internal_state, key, action):
        observation = jnp.concatenate([
            data.qpos[self.actuator_joint_mask_qpos],
            data.qvel[self.actuator_joint_mask_qvel],
            action,
            self.terrain_function.check_feet_floor_contact(data),
            internal_state["feet_time_on_ground"],
            internal_state["feet_time_in_air"],
            data.sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim],
            internal_state["goal_velocities"],
            internal_state["imu_orientation_rotation_inverse"].apply(jnp.array([0.0, 0.0, -1.0])),
            jnp.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            jnp.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
        ])

        # Add noise
        observation = self.observation_noise_function.modify_observation(internal_state, observation, key)

        # Normalize and clip
        observation = observation.at[self.joint_positions_obs_idx].set((observation[self.joint_positions_obs_idx] - internal_state["actuator_joint_nominal_positions"]) / 3.14)
        observation = observation.at[self.joint_velocities_obs_idx].set(observation[self.joint_velocities_obs_idx] / 100.0)
        observation = observation.at[self.joint_previous_actions_obs_idx].set(observation[self.joint_previous_actions_obs_idx] / 10.0)
        observation = observation.at[self.feet_ground_contact_obs_idx].set((observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0)
        observation = observation.at[self.feet_time_on_ground_obs_idx].set(jnp.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.feet_time_in_air_obs_idx].set(jnp.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.imu_linear_vel_obs_idx].set(jnp.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0))
        observation = observation.at[self.imu_angular_vel_obs_idx].set(jnp.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(jnp.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[self.critic_exteroception_obs_idx].set(jnp.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -10.0, 10.0)

        return observation
    

    def handle_domain_randomization(self, internal_state, mjx_model, data, key, is_episode_start=False):
        domain_sampling_key, domain_perturbation_sampling_key, seen_robot_key, unseen_robot_key, mujoco_model_key, action_delay_key, joint_dropout_key, perturbation_key = jax.random.split(key, 8)

        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup(domain_sampling_key)
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain_step = self.domain_randomization_sampling_function.step(domain_sampling_key)
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain = jnp.where(is_episode_start, should_randomize_domain_episode_start | internal_state["in_eval_mode"], should_randomize_domain_step)
        should_randomize_domain_perturbation = jnp.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

        self.domain_randomization_unseen_robot_function.sample(internal_state, should_randomize_domain, unseen_robot_key)
        mjx_model, data = self.domain_randomization_seen_robot_function.sample(internal_state, mjx_model, data, should_randomize_domain, seen_robot_key)
        mjx_model = self.domain_randomization_mujoco_model_function.sample(internal_state, mjx_model, should_randomize_domain, mujoco_model_key)
        self.domain_randomization_action_delay_function.sample(internal_state, should_randomize_domain, action_delay_key)
        mjx_model = self.joint_dropout_function.sample(internal_state, mjx_model, should_randomize_domain, joint_dropout_key)
        self.reward_function.handle_model_change(internal_state, mjx_model, should_randomize_domain)

        data = self.domain_randomization_perturbation_function.sample(internal_state, mjx_model, data, should_randomize_domain_perturbation, perturbation_key)

        return data, mjx_model
    

    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_previous_actions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.feet_ground_contact_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_on_ground_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_in_air_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.imu_linear_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_linear_velocity_sensor_dim)])
        current_observation_idx += self.imu_linear_velocity_sensor_dim
        self.imu_angular_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_angular_velocity_sensor_dim)])
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.goal_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.gravity_vector_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.policy_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations

        self.policy_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.feet_ground_contact_obs_idx,
            self.feet_time_on_ground_obs_idx,
            self.feet_time_in_air_obs_idx,
            self.imu_linear_vel_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
