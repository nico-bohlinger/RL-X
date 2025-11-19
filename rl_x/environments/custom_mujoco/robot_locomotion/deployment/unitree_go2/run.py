import os
import shutil
import struct
import json
import time
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
import jax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from policy import get_policy
from critic import get_critic


class UnitreeRemoteController:
    def __init__(self):
        # key
        self.Lx = 0           
        self.Rx = 0            
        self.Ry = 0            
        self.Ly = 0

        # button
        self.L1 = 0
        self.L2 = 0
        self.R1 = 0
        self.R2 = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.Up = 0
        self.Down = 0
        self.Left = 0
        self.Right = 0
        self.Select = 0
        self.F1 = 0
        self.F3 = 0
        self.Start = 0
       
    def parse_botton(self,data1,data2):
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    def parse_key(self,data):
        lx_offset = 4
        self.Lx = struct.unpack('<f', data[lx_offset:lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack('<f', data[rx_offset:rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack('<f', data[ry_offset:ry_offset + 4])[0]
        L2_offset = 16
        L2 = struct.unpack('<f', data[L2_offset:L2_offset + 4])[0] # Placeholder，unused
        ly_offset = 20
        self.Ly = struct.unpack('<f', data[ly_offset:ly_offset + 4])[0]


    def parse(self,remoteData):
        self.parse_key(remoteData)
        self.parse_botton(remoteData[2],remoteData[3])


class RobotHandler:
    def __init__(self):
        self.nr_actuator_joints = 12

        self.joint_positions_sim = np.zeros(self.nr_actuator_joints)
        self.joint_velocities_sim = np.zeros(self.nr_actuator_joints)
        self.orientation_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.zeros(3)

        self.control_frequency = 50.0

        self.trained_max_goal_velocity = 1.0
        self.goal_velocity_zero_clip_threshold_percentage = 0.1

        self.nominal_joint_positions = np.array([
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5
        ])

        self.lying_joint_positions = np.array([
            -0.04584759, 1.26458573, -2.79743123,
            0.03388786, 1.25516927, -2.7853148,
            -0.34251189, 1.27808392, -2.8028338,
            0.34323859, 1.27829576, -2.81149054
        ]) # measured on the real robot

        self.stand_and_lie_seconds = 1.0
        self.stand_and_lie_p_gain = 70.0
        self.stand_and_lie_d_gain = 3.0

        # Safety parameters
        self.velocity_safety_threshold = 25.0
        self.goal_velocity_max = 0.8
        self.stand_up_when_velocity_exceeded = True

        self.nn_p_gain = 20.0
        self.nn_d_gain = 0.5
        self.scaling_factor = 0.3

        self.x_goal_velocity = 0.0
        self.y_goal_velocity = 0.0
        self.yaw_goal_velocity = 0.0

        self.previous_control_mode = None
        self.last_seen_control_mode = None
        self.control_mode = None
        
        self.remote_controller = UnitreeRemoteController()

        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()

        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_subscriber.Init(self.low_state_callback, 10)

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(5.0)
        self.sport_client.Init()

        self.motion_switcher_client = MotionSwitcherClient()
        self.motion_switcher_client.SetTimeout(5.0)
        self.motion_switcher_client.Init()

        status, result = self.motion_switcher_client.CheckMode()
        while result['name']:
            self.sport_client.StandDown()
            self.motion_switcher_client.ReleaseMode()
            status, result = self.motion_switcher_client.CheckMode()
            time.sleep(1)

        self.default_low_cmd = unitree_go_msg_dds__LowCmd_()
        self.default_low_cmd.head[0] = 0xFE
        self.default_low_cmd.head[1] = 0xEF
        self.default_low_cmd.level_flag = 0xFF
        self.default_low_cmd.gpio = 0
        for i in range(20):
            self.default_low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.default_low_cmd.motor_cmd[i].q = 0.0 # 2.146e9
            self.default_low_cmd.motor_cmd[i].dq = 0.0 # 16000.0
            self.default_low_cmd.motor_cmd[i].kp = 0.0
            self.default_low_cmd.motor_cmd[i].kd = 0.0
            self.default_low_cmd.motor_cmd[i].tau = 0.0
        
        self.crc = CRC()

        # jax.config.update("jax_platform_name", "cpu")
        self.load_policy("latest.model")

        self.low_cmd_write_thread = RecurrentThread(interval=1/self.control_frequency, target=self.timer_callback, name="writebasiccmd")
        self.low_cmd_write_thread.Start()

        print(f"Robot ready. Model running on {jax.default_backend()}.")


    def switch_control_mode(self, control_mode):
        self.previous_control_mode = self.control_mode if self.control_mode != control_mode else self.previous_control_mode
        self.control_mode = control_mode
    

    def low_state_callback(self, msg):
        motor_states = msg.motor_state
        self.joint_positions = np.array([motor_states[i].q for i in range(self.nr_actuator_joints)])

        self.joint_velocities = np.array([motor_states[i].dq for i in range(self.nr_actuator_joints)])

        self.orientation_xyzw = np.array([msg.imu_state.quaternion[1], msg.imu_state.quaternion[2], msg.imu_state.quaternion[3], msg.imu_state.quaternion[0]])
        self.angular_velocity = np.array(msg.imu_state.gyroscope)
        
        self.remote_controller.parse(msg.wireless_remote)
        
        self.x_goal_velocity = np.clip(self.remote_controller.Ly, -1.0, 1.0) * self.goal_velocity_max
        self.y_goal_velocity = np.clip(-self.remote_controller.Lx, -1.0, 1.0) * self.goal_velocity_max
        self.yaw_goal_velocity = np.clip(-self.remote_controller.Rx, -1.0, 1.0) * self.goal_velocity_max

        if self.remote_controller.Y == 1:
            self.switch_control_mode("stand_up")
        elif self.remote_controller.B == 1:
            self.switch_control_mode("nn")
        elif self.remote_controller.X == 1:
            self.switch_control_mode("lie_down")
        elif self.remote_controller.A == 1:
            self.switch_control_mode("stop")
    

    def timer_callback(self):
        if np.max(np.abs(self.joint_velocities)) > self.velocity_safety_threshold:
            print("Velocity safety threshold exceeded.")
            if self.stand_up_when_velocity_exceeded:
                self.switch_control_mode("stand_up")

        if self.control_mode == "stand_up":
            self.stand_up()
        elif self.control_mode == "lie_down":
            self.lie_down()
        elif self.control_mode == "nn":
            self.nn()
        elif self.control_mode == "stop":
            ...
        
        self.last_seen_control_mode = self.control_mode

    
    def stand_up(self):
        if self.last_seen_control_mode != "stand_up":
            self.standing_delta = self.nominal_joint_positions - self.joint_positions
            self.standing_intermediate_position = deepcopy(self.joint_positions)
            self.standing_counter = 0
        
        if self.standing_counter < self.stand_and_lie_seconds * self.control_frequency:
            self.standing_counter += 1
            self.standing_intermediate_position += self.standing_delta / (self.stand_and_lie_seconds * self.control_frequency)
        else:
            target_positions = self.nominal_joint_positions

        low_cmd = deepcopy(self.default_low_cmd)
        for i in range(self.nr_actuator_joints):
            low_cmd.motor_cmd[i].q = target_positions[i]
            low_cmd.motor_cmd[i].kp = self.stand_and_lie_p_gain
            low_cmd.motor_cmd[i].kd = self.stand_and_lie_d_gain
        
        low_cmd.crc = self.crc.Crc(low_cmd)

        self.low_cmd_publisher.Write(low_cmd)

    
    def lie_down(self):
        if self.last_seen_control_mode != "lie_down":
            self.lying_delta = self.lying_joint_positions - self.joint_positions
            self.lying_intermediate_position = deepcopy(self.joint_positions)
            self.lying_counter = 0
        
        if self.lying_counter < self.stand_and_lie_seconds * self.control_frequency:
            self.lying_counter += 1
            self.lying_intermediate_position += self.lying_delta / (self.stand_and_lie_seconds * self.control_frequency)
        else:
            target_positions = self.lying_joint_positions

        low_cmd = deepcopy(self.default_low_cmd)
        for i in range(self.nr_actuator_joints):
            low_cmd.motor_cmd[i].q = target_positions[i]
            low_cmd.motor_cmd[i].kp = self.stand_and_lie_p_gain
            low_cmd.motor_cmd[i].kd = self.stand_and_lie_d_gain
        
        low_cmd.crc = self.crc.Crc(low_cmd)

        self.low_cmd_publisher.Write(low_cmd)
    

    def nn(self):
        # Only run neural network if it's already running or if the robot is standing up
        if self.last_seen_control_mode != "nn" and self.last_seen_control_mode != "stand_up":
            return
        
        if self.last_seen_control_mode != "nn":
            self.previous_action = np.zeros(self.nr_actuator_joints)

        goal_velocities = np.array([self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity])
        goal_velocities = np.where(np.abs(goal_velocities) < (self.goal_velocity_zero_clip_threshold_percentage * self.trained_max_goal_velocity), 0.0, goal_velocities)

        orientation_quat_inv = R.from_quat(self.orientation_xyzw).inv()
        projected_gravity_vector = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))

        observation = np.concatenate([
            (self.joint_positions - self.nominal_joint_positions) / 3.14,
            self.joint_velocities / 100.0,
            self.previous_action / 10.0,
            np.clip(self.angular_velocity / 50.0, -1.0, 1.0),
            goal_velocities,
            projected_gravity_vector
        ])
        observation = np.clip(np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
        
        action, _ = self.policy.apply(self.policy_state.params, observation)
        action = jax.device_get(action[0])

        target_joint_positions = self.nominal_joint_positions + self.scaling_factor * action

        low_cmd = deepcopy(self.default_low_cmd)
        for i in range(self.nr_actuator_joints):
            low_cmd.motor_cmd[i].q = target_joint_positions[i]
            low_cmd.motor_cmd[i].kp = self.nn_p_gain
            low_cmd.motor_cmd[i].kd = self.nn_d_gain
        
        low_cmd.crc = self.crc.Crc(low_cmd)

        self.low_cmd_publisher.Write(low_cmd)
        
        self.previous_action = action
    

    def load_policy(self, jax_model_file_name):
        current_path = os.path.dirname(__file__)
        checkpoint_dir = os.path.join(current_path, "policies")
        jax_model_path = os.path.join(checkpoint_dir, jax_model_file_name)

        shutil.unpack_archive(jax_model_path, f"{checkpoint_dir}/tmp", "zip")
        unpacked_checkpoint_dir = f"{checkpoint_dir}/tmp"

        algorithm_config = json.load(open(f"{unpacked_checkpoint_dir}/config_algorithm.json", "r"))

        self.policy = get_policy(algorithm_config)
        critic = get_critic(algorithm_config)

        self.policy.apply = jax.jit(self.policy.apply)

        key = jax.random.PRNGKey(0)
        key, policy_key, critic_key = jax.random.split(key, 3)

        dummy_policy_observation = np.zeros((1, 45))
        dummy_critic_observation = np.zeros((1, 61))

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_policy_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(0.0),
                optax.inject_hyperparams(optax.adam)(learning_rate=lambda count: 0.0),
            )
        )

        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(critic_key, dummy_critic_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(0.0),
                optax.inject_hyperparams(optax.adam)(learning_rate=lambda count: 0.0),
            )
        )

        target = {
            "policy": self.policy_state,
            "critic": critic_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(unpacked_checkpoint_dir, item=target, restore_args=restore_args)
        self.policy_state = checkpoint["policy"]

        self.policy.apply(self.policy_state.params, dummy_policy_observation)  # call to jit compile

        shutil.rmtree(unpacked_checkpoint_dir)


def main(args=None):
    robot_handler = RobotHandler()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
