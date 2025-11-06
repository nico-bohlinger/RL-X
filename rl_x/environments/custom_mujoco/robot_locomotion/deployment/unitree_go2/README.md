# Deployment on Unitree Go2 Robot

Contains the script to deploy a trained policy on the Unitree Go2 robot.

> ⚠️ **Warning**: Be careful when testing on the real robot, make sure you stay at a safe distance. The robot might behave unexpectedly. We do not take any responsibility for damages caused during deployment.

## Requirements
Install [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) to interface with the robot.

## Usage
1. Drop your trained policy checkpoint in the `policies/` directory.
2. In ```run.py```, replace  ```self.load_policy("latest.model")``` with your checkpoint filename.
3. In ```run.py```, set the safety parameters according to your needs.
4. Start both the robot and the joystick controller with a short and long press of their power buttons.
5. Put the robot on the ground in dampening mode. Depending on the robot's firmware, this might be done by pressing the L2+A buttons twice. 
6. Run the script with Python on your laptop (connected via ethernet to the robot) or directly on the robot:
   ```bash
   python3 run.py
   ```
7. Let the robot stand up by pressing the **Y** button on the joystick.
8. Start the policy by pressing the **B** button on the joystick. Be ready to cancel the script with Ctrl+C at any time if the robot behaves unexpectedly.
9. Control the robot with the joystick.
10. To stop the policy, press the **Y** button again to make the robot stand and then press the **X** button to make it lie down.
11. Stop the script with Ctrl+C.
12. Turn off the robot and the joystick controller with a short and long press of their power buttons.

### Controls
- **Y**: Stand up
- **X**: Lie down
- **B**: Start policy
- **A**: Noop
- **Left Joystick**: Forward/Backward and Left/Right velocity commands
- **Right Joystick**: Yaw velocity commands