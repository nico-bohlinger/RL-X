from pathlib import Path
from typing import Any, Dict
import mujoco
from mujoco import mjx
from flax import struct
import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym


@struct.dataclass
class State:
    data: mjx.Data
    observation: jax.Array
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = struct.field(default_factory=dict)


class Ant:
    def __init__(self, horizon=1000):
        self.horizon = horizon
        xml_path = (Path(__file__).resolve().parent / "data" / "ant.xml").as_posix()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.nr_intermediate_steps = 1
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.put_model(mj_model)

    def reset(self, key):
        key, subkey = jax.random.split(key)

        qpos = self.sys.qpos0
        qvel = jnp.zeros(self.sys.nv)
        data = mjx.put_data(self.model, self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)

        observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = False
        logging_info = {
            "episode_return": reward,
            "episode_length": 0,
        }
        info = {
            **logging_info,
            "final_observation": jnp.zeros_like(observation),
            "final_info": {**logging_info},
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

        next_observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = state.info["episode_length"] >= self.horizon
        state.info["episode_return"] += reward
        state.info["episode_length"] += 1

        done = terminated | truncated
        def when_done(_):
            __, reset_key = jax.random.split(state.info["key"])
            start_state = self.reset(reset_key)
            start_state.info["final_observation"] = next_observation
            info_keys_to_remove = ["key", "final_observation", "final_info"]
            start_state.info["final_info"] = {key: state.info[key] for key in state.info if key not in info_keys_to_remove}
            return start_state
        def when_not_done(_):
            return state.replace(data=data, observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state

    def get_observation(self, data):
        return jnp.concatenate([data.qpos[7:], data.qvel[6:]])


class GymWrapper:
    def __init__(self, env, seed=0, nr_envs=1):
        self.env = env
        self.seed = seed
        self.nr_envs = nr_envs
        self.reset_fn = jax.jit(jax.vmap(self.env.reset))
        self.step_fn = jax.jit(jax.vmap(self.env.step))
        self.key = jax.random.PRNGKey(seed)
        action_bounds = self.env.model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=jnp.float32)
        self.observation_space = gym.spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(16,), dtype=jnp.float32)

    def reset(self):
        keys = jax.random.split(self.key, self.nr_envs+1)
        self.key, env_keys = keys[0], keys[1:]

        self.state = self.reset_fn(env_keys)
        observation = self.state.observation
        info = self.get_gym_info()

        return observation, info
    
    def step(self, action):
        self.state = self.step_fn(self.state, action)
        observation = self.state.observation
        reward = self.state.reward
        terminated = self.state.terminated
        truncated = self.state.truncated
        info = self.get_gym_info()

        return observation, reward, terminated, truncated, info
    
    def get_gym_info(self):
        info_keys_to_remove = ["key",]
        info = {key: self.state.info[key] for key in self.state.info if key not in info_keys_to_remove}

        return info


USE_GPU = True
NR_ENVS = 4000
SEED = 0
LOGGING_FREQUENCY = 100000

if not USE_GPU:
    jax.config.update("jax_platform_name", "cpu")
print(f"Using device: {jax.default_backend()}")

env = Ant()
env = GymWrapper(env, seed=SEED, nr_envs=NR_ENVS)

###
import time
previous_time = time.time()
step = 0
###

state, info = env.reset()
while True:
    action = np.random.uniform(env.action_space.low, env.action_space.high, size=(env.nr_envs, env.action_space.shape[0]))
    observation, reward, terminated, truncated, info = env.step(action)
    
    ###
    step += env.nr_envs
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second")
        previous_time = current_time
    ###