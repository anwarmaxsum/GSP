import copy
import importlib
import logging
import numpy as np
import supersuit as ss
import math
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

from gymnasium import spaces
class PettingZooAtariEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]
        del self.args["scenario"]
        self.discrete = True
        if (
            "continuous_actions" in self.args
            and self.args["continuous_actions"] == True
        ):
            self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26
        
        # if self.scenario == "simple_spread_v2":
        #     self.args["N"] = 5
        
        #/home/anwar/anaconda3/envs/mambsk/lib/python3.10/site-packages/multi_agent_ale_py/roms/entombed.bin
        #/home/anwar/anaconda3/envs/mambsk/lib/python3.10/site-packages/AutoROM/roms/et.bin
        # scenario_to_bin = {"entombed_cooperative_v3":"et.bin", "joust_v3":"joust.bin",
        # # scenario_to_bin = {"entombed_cooperative_v3":"etombed.bin", "joust_v3":"joust.bin",
        #                     "mario_bros_v3":"mario_bros.bin", "wizard_of_wor_v3":"wizard_of_wor.bon"}
        
        # # rom_path = "/home/anwar/anaconda3/envs/mambsk/lib/python3.10/site-packages/AutoROM/roms/" + scenario_to_bin[self.scenario]
        # # rom_path = "~/autorom/" + scenario_to_bin[self.scenario]
        # rom_path = "~/"
        # # rom_path = "/home/anwar/anaconda3/envs/mambsk/lib/python3.10/site-packages/multi_agent_ale_py/roms/" + scenario_to_bin[self.scenario]
        
        # # self.args["rom_path"] = rom_path
        # self.rom_path = rom_path
        # self.auto_rom_install_path=rom_path

        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.atari." + self.scenario)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )
        # print(dir(self.env))

        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        # self.share_observation_space = self.repeat(self.env.state)
        # self.share_observation_space = self.get_agents_obs(self.env.observation_spaces)
        # print("check env obs spaces ", self.env.observation_spaces)
        self.share_observation_space = self.unwrap(self.env.observation_spaces)
        # self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        # self.observation_space = self.unwrap(self.env.observation_space)
        self.action_space = self.unwrap(self.env.action_spaces)
        # self.action_space = self.unwrap(self.env.action_space)
        
        # self.longest_observation_space = spaces.Box(
        #     low=low, high=high, shape=(size,), dtype=data_type
        # )
        for i in range(len(self.share_observation_space)):
            size = math.prod(self.share_observation_space[i].shape)
            # self.share_observation_space[i].shape = spaces.Box(
            #     low=self.share_observation_space[i].low[0], high=self.share_observation_space[i].high[0], shape=(size,), 
            #     dtype=self.share_observation_space[i].dtype
            # )
            self.share_observation_space[i] = spaces.Box(0, 255, (size,), self.share_observation_space[i].dtype)
            self.observation_space[i] = spaces.Box(0, 255, (size,), self.observation_space[i].dtype)

            
        # self.share_observation_space = [o.flatten() for o in self.observation_space]
        
        # print("check box ", self.share_observation_space)
        self._seed = 0

        # print("create env success")

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.discrete:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions.flatten()))
        else:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        
        # s_obs = self.repeat(self.env.state())
        # s_obs = self.repeat(self.env.state())
        # s_obs = obs
        # s_obs = [o.flatten() for o in obs]
        
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            # self.unwrap(obs),
            [o.flatten() for o in self.unwrap(obs)],
            [o.flatten() for o in self.unwrap(obs)],
            # s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        # obs = self.unwrap(self.env.reset(seed=self._seed))
        obs = [o.flatten() for o in self.unwrap(self.env.reset(seed=self._seed))]
        # print("check obs: ", [s.shape for s in obs])
        # s_obs = self.repeat(self.env.state())
        #s_obs = self.get_agents_obs(self.env.observation_spaces)
        s_obs = obs
        # s_obs = [o.flatten() for o in obs]
        # s_obs = self.repeat(self.env.state)
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        if isinstance(d, tuple):
            d = d[0]
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def get_agents_obs(self, agent_obs):
        # print("agent_obs: ", agent_obs)
        return [agent_obs[a] for a in agent_obs.keys()]