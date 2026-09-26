import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space

import random
import copy

class StochasticPolicyMH10(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicyMH10, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.max_obs_length = args["max_obs_length"]
        self.device = device
        self.action_space = action_space
        

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = torch.nn.ModuleList([RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            ).to(self.device)]*4)


        self.act = torch.nn.ModuleList([ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        ).to(self.device)])


        self.key_length = self.args["key_length"] if "key_length" in self.args.keys() else 1
        print("agent keys: ", self.args.keys())

        # self.obs_key = [nn.Parameter(torch.zeros(self.args["max_obs_length"])).to(self.device)]
        # self.feat_key = [nn.Parameter(torch.zeros(self.hidden_sizes[-1])).to(self.device)]

        # self.obs_key = nn.ParameterList([nn.Parameter(torch.randn(self.args["max_obs_length"],requires_grad=True)).to(self.device)])
        # self.feat_key = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_sizes[-1],requires_grad=True)).to(self.device)])
        # self.rfeat_key = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_sizes[-1],requires_grad=True)).to(self.device)])

        self.gating_network = nn.Sequential(
            nn.Flatten(),  # Add this line
            nn.Linear(self.args["max_obs_length"], 4),  # Modify this line
            nn.Softmax(dim=1))

        self.obs_key = nn.ParameterList([nn.Parameter(torch.zeros(self.args["max_obs_length"],requires_grad=True)).to(self.device)])
        self.feat_key = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_sizes[-1],requires_grad=True)).to(self.device)])
        self.rfeat_key = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_sizes[-1],requires_grad=True)).to(self.device)])


        # self.obs_key = nn.Parameter(torch.randn(self.args["max_obs_length"],requires_grad=True)).to(self.device)
        # self.feat_key = nn.Parameter(torch.randn(self.hidden_sizes[-1], requires_grad=True)).to(self.device)

        # self.obs_key = [torch.randn(self.args["max_obs_length"],requires_grad=True).to(self.device)]
        # self.feat_key = [torch.randn(self.hidden_sizes[-1], requires_grad=True).to(self.device)]

        self.eval_mode =  False
        self.to(device)
        

    def pad_obs(self, x):
        # print(x.shape)
        if (self.max_obs_length > x.shape[1]):
            pad = torch.zeros(x.shape[0], self.max_obs_length-x.shape[1]).to(x.device)
            x = torch.cat((x,pad),dim=-1)
        return x 


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def add_new_headlayer(self, new_head=None):
        # print(x.shape)
        # self.act.append(
        #     ACTLayer(
        #     action_space,
        #     self.hidden_sizes[-1],
        #     self.initialization_method,
        #     self.gain,
        #     args,
        # ).to(self.device)
        #     )
        # self.rnn.append(copy.deepcopy(self.rnn[-1]))
        # self.act.append(copy.deepcopy(self.act[0]))

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     self.rnn.append(
        #         # RNNLayer(
        #         #     self.hidden_sizes[-1],
        #         #     self.hidden_sizes[-1],
        #         #     self.recurrent_n,
        #         #     self.initialization_method,
        #         # ).to(self.device)
        #         copy.deepcopy(self.rnn[-1])
        #     )

        # self.act.append(
        #     ACTLayer(
        #         # action_space,
        #         # self.args["max_act_length"],
        #         self.action_space,
        #         self.hidden_sizes[-1],
        #         self.initialization_method,
        #         self.gain,
        #         self.args,
        #         # args
        #     ).to(self.device)
        #     # copy.deepcopy(self.act[0])
        # )
        new_act =    ACTLayer(
                # action_space,
                # self.args["max_act_length"],
                self.action_space,
                self.hidden_sizes[-1],
                self.initialization_method,
                self.gain,
                self.args,
                # args
            ).to(self.device)
        new_act.load_state_dict(self.act[0].state_dict())
        self.act[0] = new_act


        for i in range(len(self.obs_key)):
            self.obs_key[i].requires_grad = False
            self.obs_key[i].requires_grad = False
            # for param in self.rnn[i].parameters():
            #     param.requires_grad = False
            # for param in self.act[i].parameters():
            #     param.requires_grad = False

        # self.obs_key.append(nn.Parameter(torch.zeros(self.args["max_obs_length"])).to(self.device))
        # self.feat_key.append(nn.Parameter(torch.zeros(self.hidden_sizes[-1])).to(self.device))
        # self.obs_key.append(nn.Parameter(torch.randn(self.args["max_obs_length"],requires_grad=True)).to(self.device))
        # self.feat_key.append(nn.Parameter(torch.randn(self.hidden_sizes[-1],requires_grad=True)).to(self.device))
        # self.rfeat_key.append(nn.Parameter(torch.randn(self.hidden_sizes[-1],requires_grad=True)).to(self.device))

        self.obs_key.append(nn.Parameter(torch.zeros(self.args["max_obs_length"],requires_grad=True)).to(self.device))
        self.feat_key.append(nn.Parameter(torch.zeros(self.hidden_sizes[-1],requires_grad=True)).to(self.device))
        self.rfeat_key.append(nn.Parameter(torch.zeros(self.hidden_sizes[-1],requires_grad=True)).to(self.device))

        # for param in self.rnn[len(self.obs_key)-2].parameters():
        #     param.requires_grad = False

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        obs = self.pad_obs(obs)
        weights = self.gating_network(obs)
        # print("check weights shape head: ", weights.shape)

        actor_features = self.base(obs)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)-1
        # print("check weights: ",weights[:,0].unsqueeze(-1).shape, weights )
            
        # print("predicted head: ", idx)
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            # idx, actor_features, rnn_states = self.predict_head_idx_rf(obs, actor_features, rnn_states, masks)
            a_features, r_states =[], []
            for i in range(len(self.rnn)):
               feat, state   = self.rnn[i](actor_features, rnn_states, masks)
               # print("check feat: ",feat.shape)
               w =  weights[:,i].unsqueeze(-1)
               a_features.append(feat*w)
               r_states.append(state*w.unsqueeze(-1))
            
            # print("check w: ",w.shape)
            # print("FWD feat and state shape: ", actor_features[0].shape, r_states[0].shape)
            # print("FWD feat and state shape: ", actor_features[0].shape, r_states[1].shape)

            a_features = torch.stack(a_features).sum(dim=0)
            r_states = torch.stack(r_states).sum(dim=0)
            # print("FWD feat and state shape: ", actor_features.shape, state.shape)
            actor_features = a_features
            rnn_states = r_states
            # print("feat and state shape: ", actor_features.shape, r_states.shape)

        actions, action_log_probs = self.act[0](
            actor_features, available_actions, deterministic
        )



        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dis t_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        obs = self.pad_obs(obs)

        weights = self.gating_network(obs)

        actor_features = self.base(obs)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

  
      

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            # actor_features, rnn_states = self.rnn[idx](actor_features, rnn_states, masks)

            a_features, r_states =[], []
            for i in range(len(self.rnn)):
                feat, state   = self.rnn[i](actor_features, rnn_states, masks)
                w =  weights[:,i].unsqueeze(-1)
                a_features.append(feat*w)
                r_states.append(state*w)
                
            a_features = torch.stack(a_features).sum(dim=0)
            r_states = torch.stack(r_states).sum(dim=0)
            actor_features = a_features
            rnn_states = r_states
            # print("EVA feat and state shape: ", actor_features.shape, rnn_states.shape)

        action_log_probs, dist_entropy, action_distribution = self.act[0].evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )


        return action_log_probs, dist_entropy, action_distribution


    def predict_head_idx(self, obs, feat):
        head_sim = []
        for i in range(len(self.obs_key)):
            obs_norm = self.l2_normalize(obs, dim=-1) 
            obs_key_norm = self.l2_normalize(self.obs_key[i], dim=-1) 
            feat_norm = self.l2_normalize(feat, dim=-1) 
            feat_key_norm = self.l2_normalize(self.feat_key[i], dim=-1)

            # obs_norm = obs
            # obs_key_norm = self.obs_key[i] 
            # feat_norm = feat 
            # feat_key_norm = self.feat_key[i]

            sim = obs_key_norm * obs_norm # B, top_k, C
            tsim = torch.sum(sim) / obs.shape[0] # Scalar

            sim2 = feat_key_norm * feat_norm # B, top_k, C
            tsim2 = torch.sum(sim2) / feat.shape[0] # Scalar

            if tsim < 0 and tsim2 < 0:
                tsim2 = tsim2 * -1.0

            head_sim.append(100* tsim * tsim2)

        # print("head sim: ", torch.stack(head_sim), end=" \n")
        return torch.argmax(torch.stack(head_sim))


    def predict_head_idx_rf(self, obs, feat, rnn_states, masks):
        head_sim = []
        rfeat_list = []
        rnn_states_list = []
        tsim_list = []
        tsim2_list = []
        tsim3_list = []

        for i in range(len(self.rfeat_key)):
            # rfeat = self.rnn[i](feat)
            rfeat, ornn_states = self.rnn[i](feat, rnn_states, masks)
            rfeat_list.append(rfeat)
            rnn_states_list.append(ornn_states)
            rfeat_norm = self.l2_normalize(rfeat, dim=-1) 
            rfeat_key_norm = self.l2_normalize(self.rfeat_key[i], dim=-1)

            sim = rfeat_key_norm * rfeat_norm # B, top_k, C
            tsim = torch.sum(sim) / feat.shape[0] # Scalar
            

            obs_norm = self.l2_normalize(obs, dim=-1) 
            obs_key_norm = self.l2_normalize(self.obs_key[i], dim=-1) 
            sim2 = obs_key_norm * obs_norm # B, top_k, C
            tsim2 = torch.sum(sim2) / obs.shape[0] # Scalar

            feat_norm = self.l2_normalize(feat, dim=-1) 
            feat_key_norm = self.l2_normalize(self.feat_key[i], dim=-1) 
            sim3 = feat_key_norm * feat_norm # B, top_k, C
            tsim3 = torch.sum(sim3) / feat.shape[0] # Scalar

            tsim_list.append(tsim)
            tsim2_list.append(tsim2)
            tsim3_list.append(tsim3)

            
            # if tsim < 0 and tsim2 < 0:
            #     tsim2 = tsim2 * -1.0

            # head_sim.append(100* tsim * tsim2)
  
            # head_sim.append(tsim)
        # tsim_list = torch.stack(tsim_list)
        # tsim2_list = torch.stack(tsim2_list)
        # tsim3_list = torch.stack(tsim3_list)

        # sofmax = nn.Softmax(dim=0)
        # tsim_list = sofmax(tsim_list)
        # tsim2_list = sofmax(tsim2_list)
        # tsim3_list = sofmax(tsim3_list)

        # head_sim = tsim_list * tsim2_list * tsim3_list
        # # head_sim = tsim_list 
        # # print("head sim: ", torch.stack(head_sim), end=" \n")
        # # idx  = torch.argmax(torch.stack(head_sim))
        # idx  = torch.argmax(head_sim)
        # return idx, rfeat_list[idx], rnn_states_list[idx]


        tsim_list = torch.stack(tsim_list)
        tsim2_list = torch.stack(tsim2_list)
        tsim3_list = torch.stack(tsim3_list)

        sofmax = nn.Softmax(dim=0)
        tsim_list = sofmax(tsim_list)
        tsim2_list = sofmax(tsim2_list)
        tsim3_list = sofmax(tsim3_list)


        # head_sim = tsim2_list*tsim_list*10000
        head_sim = tsim2_list*10000


        idx  = torch.argmax(head_sim)

        # print("head sim: ", head_sim, "chosen idx: ",idx)
        return idx, rfeat_list[idx], rnn_states_list[idx]



    def evaluate_actions_for_training(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        
        obs = self.pad_obs(obs)
        weights = self.gating_network(obs)

        actor_features = self.base(obs)


        if self.use_naive_recurrent_policy or self.use_recurrent_policy:

            a_features, r_states =[], []
            for i in range(len(self.rnn)):
                feat, state   = self.rnn[i](actor_features, rnn_states, masks)
                w =  weights[:,i].unsqueeze(-1)
                a_features.append(feat*w)
                r_states.append(state*w)
            

            a_features = torch.stack(a_features).sum(dim=0)
            r_states = torch.stack(r_states).sum(dim=0)
            actor_features = a_features
            rnn_states = r_states

        
        action_log_probs, dist_entropy, action_distribution = self.act[0].evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )

        # return action_log_probs, dist_entropy, action_distribution, extra_loss, actor_features
        return action_log_probs, dist_entropy, action_distribution, obs, actor_features



    def forward_for_key_sim(self, obs, actor_features, rnn_states, masks):



        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        obs = self.pad_obs(obs)
        actor_features = self.base(obs)
        actor_rfeat, ornn_states  = self.rnn[-1](actor_features,rnn_states, masks)
        sim_loss = {}

        # if self.training:
        obs_norm = self.l2_normalize(obs, dim=-1) 
        obs_key_norm = self.l2_normalize(self.obs_key[-1], dim=-1) 
        feat_norm = self.l2_normalize(actor_features, dim=-1) 
        feat_key_norm = self.l2_normalize(self.feat_key[-1], dim=-1)

        rfeat_norm = self.l2_normalize(actor_rfeat, dim=-1) 
        rfeat_key_norm = self.l2_normalize(self.rfeat_key[-1], dim=-1)


        # obs_norm = obs
        # obs_key_norm = self.obs_key[-1] 
        # feat_norm = actor_features
        # feat_key_norm = self.feat_key[-1]


        sim = obs_key_norm * obs_norm # B, top_k, C
        tsim = torch.sum(sim) / obs.shape[0] # Scalar
        sim_loss['obs_sim'] = tsim

        sim2 = feat_key_norm * feat_norm # B, top_k, C
        tsim2 = torch.sum(sim2) / actor_features.shape[0] # Scalar
        sim_loss['feat_sim'] = tsim2

        sim3 = rfeat_key_norm * rfeat_norm # B, top_k, C
        tsim3 = torch.sum(sim3) / actor_features.shape[0] # Scalar
        sim_loss['rfeat_sim'] = tsim3


        # extra_loss['feat_sim'] = torch.zeros((1),requires_grad=True)
        # extra_loss['obs_sim'] = torch.zeros((1),requires_grad=True)
        
        return sim_loss