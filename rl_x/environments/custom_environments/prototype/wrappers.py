import time
import logging
import select as sl
import gymnasium as gym
import numpy as np

from rl_x.environments.vec_env import VecEnvWrapper, SubprocVecEnv, _flatten_obs

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType

rlx_logger = logging.getLogger("rl_x")


class RecordEpisodeStatistics(VecEnvWrapper):
    def __init__(self, venv):
        super(RecordEpisodeStatistics, self).__init__(venv)
        self.num_envs = getattr(venv, "num_envs", 1)
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self):
        observations = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step_wait(self):
        observations, rewards, terminations, truncatations, infos = self.venv.step_wait()
        dones = terminations | truncatations
        self.episode_returns += rewards
        self.episode_lengths += 1
        for i in range(len(dones)):
            if dones[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length
                }
                infos[i]["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return observations, rewards, terminations, truncatations, infos


class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)

    def reset(self):
        return self.env.reset()

    def get_episode_infos(self, info):
        episode_infos = []
        for single_info in info:
            maybe_episode_info = single_info.get("episode")
            if maybe_episode_info is not None:
                episode_infos.append(maybe_episode_info)
        return episode_infos
    

    def get_final_observation(self, info, id):
        return info[id]["final_observation"]

    
    def get_action_space_type(self):
        return ActionSpaceType.CONTINUOUS


    def get_single_action_space_shape(self):
        return self.action_space.shape


    def get_observation_space_type(self):
        return ObservationSpaceType.FLAT_VALUES


class NonblockingVecEnv(SubprocVecEnv):
    threshold = 0

    def __init__(self, env_fns, synchronous_steps, sync_threshold, start_method=None):
        SubprocVecEnv.__init__(self, env_fns, start_method)
        self.invented = [False] * len(env_fns)
        self.threshold = len(self.remotes) * sync_threshold
        self.synchronous_steps = synchronous_steps

    def step_async(self, actions):
        i = 0
        for remote, action in zip(self.remotes, actions):
            if not self.invented[i]:
                remote.send(('step', action))
            i += 1
        self.waiting = True

    def step_wait(self):
        all_invented = True

        if self.synchronous_steps is False:
            readable, _, _ = sl.select(self.remotes,[],[])
            readable_count = len(readable)

            while readable_count < self.threshold:
                not_ready = [x for x in self.remotes if x not in readable]
                r,_,_ = sl.select(not_ready,[],[])
                readable += r
                readable_count = len(readable)

        counter = 0
        while all_invented:
            results = []
            i = 0
            for remote in self.remotes:
                if self.synchronous_steps:
                    answer = remote.recv()
                else:
                    answer = None
                    while remote.poll():
                        answer = remote.recv()

                if answer is not None:
                    results.append(answer)
                    self.invented[i] = False
                    all_invented = False
                    counter += 1
                else:
                    answer = ([0.0] * self.observation_space.shape[0], 0.0, False, False, {}, {})
                    results.append(answer)
                    self.invented[i] = True
                i += 1

            if all_invented:
                time.sleep(0.02)
                rlx_logger.warning("All invented")

        self.waiting = False
        obs, rews, terminations, truncations, infos, _ = zip(*results)

        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(terminations), np.stack(truncations), infos
