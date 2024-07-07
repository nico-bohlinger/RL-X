from pathlib import Path
import mujoco
from mujoco import mjx
from functools import partial
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.brax.state import State


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



    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        return self._reset(key)


    # @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, key):
        key, subkey = jax.random.split(key)

        data = mjx.put_data(self.model, self.data)
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel, ctrl=jnp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)

        observation = self.get_observation(data)
        reward = 0.0
        done = False
        logging_info = {
            "episode_return": reward,
            "episode_length": 0,
            "forward_vel": 0.0
        }
        info = {
            **logging_info,
            "done": False,
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
        terminated = False
        truncated = state.info["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info["episode_return"] += reward
        state.info["done"] = done

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
        observation = jnp.concatenate([
            data.qpos[2:], data.qvel
        ])

        return observation
    
    
    def get_reward(self, data):
        x_vel = data.qvel[0]

        reward = x_vel

        info = {
            "forward_vel": x_vel,
        }

        return reward, info
