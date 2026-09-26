import copy
import importlib
import logging
import numpy as np
import supersuit as ss

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class PettingZooMPEEnv2:
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

        self.cur_step = 0
        # self.module = importlib.import_module("from mpe2 import " + self.scenario)
        self.module = importlib.import_module("mpe2." + self.scenario)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )
        print(dir(self.env))
        # self.env = self.module.env()

        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        print("check self.agents: ",self.agents)

        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        # self.share_observation_space = self.repeat(self.env.state_space)
        # self.observation_space = self.unwrap(self.env.observation_space)
        # self.action_space = self.unwrap(self.env.action_space)
        self._seed = 0
        # print("shared obs: ",self.share_observation_space)

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
        s_obs = self.repeat(self.env.state())
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(seed=self._seed))
        s_obs = self.repeat(self.env.state())
        # print("check s_obs: ", [s.shape for s in s_obs])
        # print("check obs: ", [s.shape for s in obs])
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
        # print("check self agents: ",self.agents)
        # print("check d: ", d)
        idx = 0
        # d = d[0]
        if isinstance(d, tuple):
            d = d[0]
        for agent in self.agents:
            # print("check idx=",idx, " agent=",agent)
            # idx+=1
            l.append(d[agent])

        # print("finish unwrap")
        
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
