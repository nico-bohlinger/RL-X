def environment_creator(config):

    import gymnasium as gym
    import torch
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
    import isaacsim.core.utils.torch as torch_utils
    from isaaclab.assets import Articulation


    ANT_CFG = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Ant/ant_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*_leg": 0.0,
                "front_left_foot": 0.785398,
                "front_right_foot": -0.785398,
                "left_back_foot": -0.785398,
                "right_back_foot": 0.785398,
            },
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


    @configclass
    class AntEnvCfg(DirectRLEnvCfg):
        episode_length_s = 15.0
        decimation = 2
        action_scale = 0.5
        action_space = 8
        observation_space = 36
        state_space = 0

        sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )

        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
        )

        robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]

        heading_weight: float = 0.5
        up_weight: float = 0.1

        energy_cost_scale: float = 0.05
        actions_cost_scale: float = 0.005
        alive_reward_scale: float = 0.5
        dof_vel_scale: float = 0.2

        death_cost: float = -2.0
        termination_height: float = 0.31

        angular_velocity_scale: float = 1.0
        contact_force_scale: float = 0.1



    class Ant(DirectRLEnv):
        cfg: DirectRLEnvCfg


        def __init__(self, config):
            super().__init__(AntEnvCfg(), config.environment.rendering_mode)

            self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=float)
            self.single_observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(36,), dtype=float)

            self.action_scale = self.cfg.action_scale
            self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
            self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
            self._joint_dof_idx, _ = self.robot.find_joints(".*")

            self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
            self.prev_potentials = torch.zeros_like(self.potentials)
            self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
                (self.num_envs, 1)
            )
            self.targets += self.scene.env_origins
            self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
            self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
            self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
                (self.num_envs, 1)
            )
            self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
            self.basis_vec0 = self.heading_vec.clone()
            self.basis_vec1 = self.up_vec.clone()

            self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
            self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
            self.last_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
            self.last_episode_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)


        def _setup_scene(self):
            self.robot = Articulation(self.cfg.robot)
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            self.scene.clone_environments(copy_from_source=False)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            self.scene.articulations["robot"] = self.robot
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)


        def _pre_physics_step(self, actions: torch.Tensor):
            self.actions = actions.clone()


        def _apply_action(self):
            forces = self.action_scale * self.joint_gears * self.actions
            self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
        

        def _compute_intermediate_values(self):
            self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
            self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
            self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

            (
                self.up_proj,
                self.heading_proj,
                self.up_vec,
                self.heading_vec,
                self.vel_loc,
                self.angvel_loc,
                self.roll,
                self.pitch,
                self.yaw,
                self.angle_to_target,
                self.dof_pos_scaled,
                self.prev_potentials,
                self.potentials,
            ) = compute_intermediate_values(
                self.targets,
                self.torso_position,
                self.torso_rotation,
                self.velocity,
                self.ang_velocity,
                self.dof_pos,
                self.robot.data.soft_joint_pos_limits[0, :, 0],
                self.robot.data.soft_joint_pos_limits[0, :, 1],
                self.inv_start_rot,
                self.basis_vec0,
                self.basis_vec1,
                self.potentials,
                self.prev_potentials,
                self.cfg.sim.dt,
            )


        def _get_observations(self) -> dict:
            obs = torch.cat(
                (
                    self.torso_position[:, 2].view(-1, 1),
                    self.vel_loc,
                    self.angvel_loc * self.cfg.angular_velocity_scale,
                    normalize_angle(self.yaw).unsqueeze(-1),
                    normalize_angle(self.roll).unsqueeze(-1),
                    normalize_angle(self.angle_to_target).unsqueeze(-1),
                    self.up_proj.unsqueeze(-1),
                    self.heading_proj.unsqueeze(-1),
                    self.dof_pos_scaled,
                    self.dof_vel * self.cfg.dof_vel_scale,
                    self.actions,
                ),
                dim=-1,
            )
            observations = {"policy": obs}
            return observations


        def _get_rewards(self) -> torch.Tensor:
            total_reward = compute_rewards(
                self.actions,
                self.reset_terminated,
                self.cfg.up_weight,
                self.cfg.heading_weight,
                self.heading_proj,
                self.up_proj,
                self.dof_vel,
                self.dof_pos_scaled,
                self.potentials,
                self.prev_potentials,
                self.cfg.actions_cost_scale,
                self.cfg.energy_cost_scale,
                self.cfg.dof_vel_scale,
                self.cfg.death_cost,
                self.cfg.alive_reward_scale,
                self.motor_effort_ratio,
            )
            self.episode_returns += total_reward
            self.episode_lengths += 1
            return total_reward


        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            self._compute_intermediate_values()
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            died = self.torso_position[:, 2] < self.cfg.termination_height
            done = died | time_out
            self.last_episode_returns[done] = self.episode_returns[done]
            self.last_episode_lengths[done] = self.episode_lengths[done]
            return died, time_out


        def _reset_idx(self, env_ids: torch.Tensor | None):
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = self.robot._ALL_INDICES
            self.robot.reset(env_ids)
            super()._reset_idx(env_ids)

            joint_pos = self.robot.data.default_joint_pos[env_ids]
            joint_vel = self.robot.data.default_joint_vel[env_ids]
            default_root_state = self.robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self.scene.env_origins[env_ids]

            self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            to_target = self.targets[env_ids] - default_root_state[:, :3]
            to_target[:, 2] = 0.0
            self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

            self._compute_intermediate_values()

            self.episode_returns[env_ids] = 0.0
            self.episode_lengths[env_ids] = 0.0


        def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
            observation, reward, terminated, truncated, info = super().step(action)
            info["episode_return"] = self.last_episode_returns.cpu().numpy()
            info["episode_length"] = self.last_episode_lengths.cpu().numpy()
            return observation["policy"], reward, terminated, truncated, info


        def reset(self) -> tuple[torch.Tensor, dict]:
            observation, info = super().reset()
            info["episode_return"] = self.last_episode_returns.cpu().numpy()
            info["episode_length"] = self.last_episode_lengths.cpu().numpy()
            return observation["policy"], info


        def close(self):
            super().close()
        

        def get_logging_info_dict(self, info: dict) -> dict:
            return info
    

    def normalize_angle(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))
    

    @torch.jit.script
    def compute_rewards(
        actions: torch.Tensor,
        reset_terminated: torch.Tensor,
        up_weight: float,
        heading_weight: float,
        heading_proj: torch.Tensor,
        up_proj: torch.Tensor,
        dof_vel: torch.Tensor,
        dof_pos_scaled: torch.Tensor,
        potentials: torch.Tensor,
        prev_potentials: torch.Tensor,
        actions_cost_scale: float,
        energy_cost_scale: float,
        dof_vel_scale: float,
        death_cost: float,
        alive_reward_scale: float,
        motor_effort_ratio: torch.Tensor,
    ) -> torch.Tensor:
        heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
        heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

        up_reward = torch.zeros_like(heading_reward)
        up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

        actions_cost = torch.sum(actions**2, dim=-1)
        electricity_cost = torch.sum(
            torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
            dim=-1,
        )

        dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

        alive_reward = torch.ones_like(potentials) * alive_reward_scale
        progress_reward = potentials - prev_potentials

        total_reward = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - actions_cost_scale * actions_cost
            - energy_cost_scale * electricity_cost
            - dof_at_limit_cost
        )
        total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
        return total_reward


    @torch.jit.script
    def compute_intermediate_values(
        targets: torch.Tensor,
        torso_position: torch.Tensor,
        torso_rotation: torch.Tensor,
        velocity: torch.Tensor,
        ang_velocity: torch.Tensor,
        dof_pos: torch.Tensor,
        dof_lower_limits: torch.Tensor,
        dof_upper_limits: torch.Tensor,
        inv_start_rot: torch.Tensor,
        basis_vec0: torch.Tensor,
        basis_vec1: torch.Tensor,
        potentials: torch.Tensor,
        prev_potentials: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        to_target = targets - torso_position
        to_target[:, 2] = 0.0

        torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
            torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
        )

        vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
            torso_quat, velocity, ang_velocity, targets, torso_position
        )

        dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

        to_target = targets - torso_position
        to_target[:, 2] = 0.0
        prev_potentials[:] = potentials
        potentials = -torch.norm(to_target, p=2, dim=-1) / dt

        return (
            up_proj,
            heading_proj,
            up_vec,
            heading_vec,
            vel_loc,
            angvel_loc,
            roll,
            pitch,
            yaw,
            angle_to_target,
            dof_pos_scaled,
            prev_potentials,
            potentials,
        )


    return Ant(config)
