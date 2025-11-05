import multiprocessing as mp
from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium.error import AlreadyPendingCallError, NoAsyncCallError
from gymnasium.vector.utils import iterate
from gymnasium.vector.async_vector_env import AsyncState


class AsyncVectorEnvWithSkipping(gym.vector.AsyncVectorEnv):
    def __init__(self, env_fns, async_skip_percentage=0.0,
                 observation_space=None, action_space=None, shared_memory=True, copy=True, start_method=None, daemon=True, worker=None):
        
        if start_method is None:
            all_start_methods = mp.get_all_start_methods()
            start_method = mp.get_start_method()
            # Only use fork if it is the only available start method to prevent interference with JAX
            if start_method == "fork":
                if "forkserver" in all_start_methods:
                    start_method = "forkserver"
                elif "spawn" in all_start_methods:
                    start_method = "spawn"
        super().__init__(env_fns, observation_space, action_space, shared_memory, copy, start_method, daemon, worker)
        
        if not shared_memory:
            raise NotImplementedError("AsyncVectorEnvWithSkipping only supports shared_memory=True.")
        
        self.nr_envs = len(self.env_fns)
        self.skipped_envs = [False] * self.nr_envs

        self.skip_threshold = int(self.nr_envs * async_skip_percentage)


    def step_async(self, actions):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.", self._state.value)

        actions = iterate(self.action_space, actions)
        for pipe, action, skipped in zip(self.parent_pipes, actions, self.skipped_envs):
            if not skipped:
                pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    
    def step_wait(self, timeout=None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError("Calling `step_wait` without any prior call " "to `step_async`.", AsyncState.WAITING_STEP.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(f"The call to `step_wait` has timed out after {timeout} second(s).")

        rewards = np.zeros(self.nr_envs)
        terminateds = np.zeros(self.nr_envs, dtype=np.bool_)
        truncateds = np.zeros(self.nr_envs, dtype=np.bool_)
        infos = {}
        successes = []

        self.skipped_envs = [True] * self.nr_envs
        while True:
            for i, pipe in enumerate(self.parent_pipes):
                if not self.skipped_envs[i]:
                    continue
                if pipe.poll():
                    result, success = pipe.recv()
                    self.skipped_envs[i] = False
                else:
                    result, success = (None, 0.0, False, False, {}), True
                    self.skipped_envs[i] = True
                successes.append(success)
                if success:
                    obs, rew, terminated, truncated, info = result

                    rewards[i] = rew
                    terminateds[i] = terminated
                    truncateds[i] = truncated
                    infos = self._add_info(infos, info, i)

            if sum(self.skipped_envs) <= self.skip_threshold:
                break

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            rewards, terminateds, truncateds, infos,
        )