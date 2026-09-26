import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.mlp import MLPLayer
from harl.models.base.rnn import RNNLayer
# from harl.models.base.act import ACTLayer
from harl.models.base.act2 import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space

import random
import copy

class StochasticPolicyMH23(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicyMH23, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.activation_func = args["activation_func"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.max_obs_length = args["max_obs_length"]
        self.device = device
        self.action_space = action_space
        
        self.n_expert = 4 # should be 4

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        
        #args["hidden_sizes"] = args["hidden_sizes"][:-1]
        self.base = base(args, obs_shape)

        self.s_experts = torch.nn.ModuleList([
            MLPLayer(self.hidden_sizes[-1], [self.hidden_sizes[-1]], self.initialization_method, self.activation_func)
            ]*self.n_expert).to(self.device)

        # self.t_experts = torch.nn.ModuleList([
        #     MLPLayer(self.hidden_sizes[-1], [self.hidden_sizes[-1]], self.initialization_method, self.activation_func)
        #     ]*self.n_expert).to(self.device)


        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            ).to(self.device)


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
            nn.Linear(self.args["max_obs_length"], self.n_expert),  # Modify this line
            nn.Softmax(dim=1))

        self.obs_key = nn.ParameterList([nn.Parameter(torch.zeros(self.args["max_obs_length"],requires_grad=True)).to(self.device)])
        self.feat_key = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_sizes[-1],requires_grad=True)).to(self.device)])
        # self.rfeat_key = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_sizes[-1],requires_grad=True)).to(self.device)])
        self.rfeat_key = nn.ParameterList([nn.Parameter(torch.zeros(self.args["max_obs_length"],requires_grad=True)).to(self.device)])

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
        new_act.load_state_dict(self.act[-1].state_dict())
        self.act.append(new_act)


        for i in range(len(self.obs_key)):
            self.obs_key[i].requires_grad = False
            # self.obs_key[i].requires_grad = False
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
        # self.rfeat_key.append(nn.Parameter(torch.zeros(self.hidden_sizes[-1],requires_grad=True)).to(self.device))
        self.rfeat_key.append(nn.Parameter(torch.zeros(self.args["max_obs_length"],requires_grad=True)).to(self.device))

        # for param in self.rnn[len(self.obs_key)-2].parameters():
        #     param.requires_grad = False

    def forward_experts(self, obs, x):
        weights = self.gating_network(obs)
        # print("check shape \n", weights.shape)
        # print((self.s_experts[0](x)).shape)
        # outputs = torch.stack([expert(x) * weight.unsqueeze() for expert, weight in zip(self.s_experts, weights)])
        a_features=[]
        for i in range(len(self.s_experts)):
           feat   = self.s_experts[i](x)
           w =  weights[:,i].unsqueeze(-1)
           a_features.append(feat*w)
        # outputs.sum(dim=0)
        return torch.stack(a_features).sum(dim=0)


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

        pert_obs  = obs + 0.01 + 0.01 * torch.rand_like(obs, device=obs.device)
        pert_obs = self.pad_obs(pert_obs)
        obs = self.pad_obs(obs)
        actor_features = self.base(obs)
        actor_features = self.forward_experts(obs,actor_features)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if len(self.act) > 1:
            # idx = self.predict_head_idx(obs, actor_features)
            idx = self.predict_head_idx(obs, pert_obs, actor_features)
        else:
            idx = 0
        # actions, action_log_probs = self.act[0](
        actions, action_log_probs = self.act[idx](
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

        pert_obs  = obs + 0.01 + 0.01 * torch.rand_like(obs, device=obs.device)
        pert_obs = self.pad_obs(pert_obs)
        obs = self.pad_obs(obs)
        actor_features = self.base(obs)
        actor_features = self.forward_experts(obs,actor_features)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if len(self.act) > 1:
            # idx = self.predict_head_idx(obs, actor_features)
            idx = self.predict_head_idx(obs, pert_obs, actor_features)
        else:
            idx = 0

        # action_log_probs, dist_entropy, action_distribution = self.act[0].evaluate_actions(
        action_log_probs, dist_entropy, action_distribution = self.act[idx].evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )


        return action_log_probs, dist_entropy, action_distribution


    #R4
    # def predict_head_idx(self, obs, feat):
    #     head_sim = []
    #     sim3_list = []
    #     mask_rfeat_key_list = []
    #     mask_rfeat_list = []

    #     for i in range(len(self.obs_key)):
    #         obs_norm = self.l2_normalize(obs, dim=-1) 
    #         obs_key_norm = self.l2_normalize(self.obs_key[i], dim=-1) 
    #         feat_norm = self.l2_normalize(feat, dim=-1) 
    #         feat_key_norm = self.l2_normalize(self.feat_key[i], dim=-1)
    #         # feat_key_norm = self.l2_normalize(self.rfeat_key[i], dim=-1)

    #         rfeat = obs 
    #         rfeat_key = self.rfeat_key[i]
    #         # rfeat_key = self.rfeat_key[i].repeat(obs.shape[0],1)

    #         # print("rfeat_key ", rfeat_key.shape)
    #         # print("obs ", obs.shape)

    #         sim = obs_key_norm * obs_norm # B, top_k, C
    #         tsim = torch.sum(sim) / obs.shape[0] # Scalar

    #         sim2 = feat_key_norm * feat_norm # B, top_k, C
    #         tsim2 = torch.sum(sim2) / feat.shape[0] # Scalar

    #         # mask_rfeat = torch.count_nonzero(torch.ceil(rfeat.mean(dim=0)))
    #         # mask_rfeat_key = torch.count_nonzero(torch.ceil(rfeat_key.mean(dim=0)))

    #         mask_rfeat = (rfeat == 0).sum() / rfeat.shape[0]
    #         # mask_rfeat_key = (rfeat_key == 0).sum() / rfeat_key.shape[0]
    #         mask_rfeat_key = (rfeat_key == 0).sum()

    #         # print("mask_rfeat_key", mask_rfeat_key)
    #         mask_rfeat_list.append(mask_rfeat)
    #         mask_rfeat_key_list.append(mask_rfeat_key)
    #         # sim3 = 10000 - torch.norm(1.0*(mask_rfeat-mask_rfeat_key))
    #         sim3 = 10000 - torch.norm(1.0*(mask_rfeat-mask_rfeat_key))
    #         sim3_list.append(sim3)
            
    #         if tsim < 0 and tsim2 < 0:
    #             tsim2 = tsim2 * -1.0

    #         head_sim.append(sim3 + (tsim * tsim2))
    #         # head_sim.append(sim3 + (tsim + tsim2))

    #     # print("head sim: ", torch.stack(head_sim), end=" \n")
    #     # if self.eval_mode:

    #     #     print("head sim: ", torch.stack(head_sim), " sim3 sim3_list", sim3_list, end=" \n")
    #     #     print("head idx: ", torch.argmax(torch.stack(head_sim)), end=" \n")
    #     #     print("mask list: ", mask_rfeat_list, " ", mask_rfeat_key_list)


    #     return torch.argmax(torch.stack(head_sim))
    

    # #R5 woth cons zeros
    # def predict_head_idx(self, obs, feat):
    #     head_sim = []
    #     sim3_list = []
    #     mask_rfeat_key_list = []
    #     mask_rfeat_list = []

    #     for i in range(len(self.obs_key)):
    #         obs_norm = self.l2_normalize(obs, dim=-1) 
    #         obs_key_norm = self.l2_normalize(self.obs_key[i], dim=-1) 
    #         feat_norm = self.l2_normalize(feat, dim=-1) 
    #         feat_key_norm = self.l2_normalize(self.feat_key[i], dim=-1)
    #         # feat_key_norm = self.l2_normalize(self.rfeat_key[i], dim=-1)

    #         rfeat = obs.mean(dim=0)
    #         rfeat_key = self.rfeat_key[i]
    #         # rfeat_key = self.rfeat_key[i].repeat(obs.shape[0],1)

    #         # print("rfeat_key ", rfeat_key.shape)
    #         # print("obs ", obs.shape)

    #         sim = obs_key_norm * obs_norm # B, top_k, C
    #         tsim = torch.sum(sim) / obs.shape[0] # Scalar

    #         sim2 = feat_key_norm * feat_norm # B, top_k, C
    #         tsim2 = torch.sum(sim2) / feat.shape[0] # Scalar


    #         mask_rfeat = self.max_consecutive_zeros(rfeat)
    #         mask_rfeat_key = self.max_consecutive_zeros(rfeat_key)

    #         # print("mask_rfeat_key", mask_rfeat_key)
    #         mask_rfeat_list.append(mask_rfeat)
    #         mask_rfeat_key_list.append(mask_rfeat_key)
    #         # sim3 = 10000 - torch.norm(1.0*(mask_rfeat-mask_rfeat_key))
    #         sim3 = 10000 - torch.norm(1.0*(mask_rfeat-mask_rfeat_key))
    #         sim3_list.append(sim3)
            
    #         if tsim < 0 and tsim2 < 0:
    #             tsim2 = tsim2 * -1.0

    #         head_sim.append(sim3 + (tsim * tsim2))
    #         # head_sim.append(sim3 + (tsim + tsim2))

        # print("head sim: ", torch.stack(head_sim), end=" \n")
        # if self.eval_mode:

        #     print("head sim: ", torch.stack(head_sim), " sim3 sim3_list", sim3_list, end=" \n")
        #     print("head idx: ", torch.argmax(torch.stack(head_sim)), end=" \n")
        #     print("mask list: ", mask_rfeat_list, " ", mask_rfeat_key_list)


    #     return torch.argmax(torch.stack(head_sim))



    #R5
    def predict_head_idx(self, obs, pert_obs, feat):
        head_sim = []
        sim3_list = []
        mask_rfeat_key_list = []
        mask_rfeat_list = []

        for i in range(len(self.obs_key)):
            obs_norm = self.l2_normalize(obs, dim=-1) 
            obs_key_norm = self.l2_normalize(self.obs_key[i], dim=-1) 
            feat_norm = self.l2_normalize(feat, dim=-1) 
            feat_key_norm = self.l2_normalize(self.feat_key[i], dim=-1)
            # feat_key_norm = self.l2_normalize(self.rfeat_key[i], dim=-1)

            rfeat = pert_obs.mean(dim=0) 
            rfeat_key = self.rfeat_key[i]
            # rfeat_key = self.rfeat_key[i].repeat(obs.shape[0],1)

            # print("rfeat_key ", rfeat_key.shape)
            # print("obs ", obs.shape)

            sim = obs_key_norm * obs_norm # B, top_k, C
            tsim = torch.sum(sim) / obs.shape[0] # Scalar

            sim2 = feat_key_norm * feat_norm # B, top_k, C
            tsim2 = torch.sum(sim2) / feat.shape[0] # Scalar


            mask_rfeat = (rfeat == 0).sum()
            mask_rfeat_key = (rfeat_key == 0).sum()

            # print("mask_rfeat_key", mask_rfeat_key)
            mask_rfeat_list.append(mask_rfeat)
            mask_rfeat_key_list.append(mask_rfeat_key)
            # sim3 = 10000 - torch.norm(1.0*(mask_rfeat-mask_rfeat_key))
            sim3 = 1000 - torch.norm(1.0*(mask_rfeat-mask_rfeat_key))
            sim3_list.append(sim3)
            
            if tsim < 0 and tsim2 < 0:
                tsim2 = tsim2 * -1.0

            head_sim.append(sim3 + (tsim * tsim2))
            # head_sim.append(sim3 + (tsim + tsim2))

        # print("head sim: ", torch.stack(head_sim), end=" \n")
        # if self.eval_mode:

        #     print("head sim: ", torch.stack(head_sim), " sim3 sim3_list", sim3_list, end=" \n")
        #     print("head idx: ", torch.argmax(torch.stack(head_sim)), end=" \n")
        #     print("mask list: ", mask_rfeat_list, " ", mask_rfeat_key_list)


        return torch.argmax(torch.stack(head_sim))
    




    def forward_for_key_sim(self, obs, actor_features, rnn_states, masks):


        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)


        pert_obs  = obs + 0.01 + 0.01 * torch.rand_like(obs, device=obs.device)
        pert_obs = self.pad_obs(pert_obs)
        obs = self.pad_obs(obs)
        actor_features = self.base(obs)
        actor_features = self.forward_experts(obs,actor_features)
        actor_rfeat, ornn_states  = self.rnn(actor_features,rnn_states, masks)
        sim_loss = {}

        # if self.training:
        obs_norm = self.l2_normalize(obs, dim=-1) 
        obs_key_norm = self.l2_normalize(self.obs_key[-1], dim=-1) 
        feat_norm = self.l2_normalize(actor_features, dim=-1) 
        feat_key_norm = self.l2_normalize(self.feat_key[-1], dim=-1)

        # rfeat_norm = self.l2_normalize(actor_rfeat, dim=-1) 
        # rfeat_key_norm = self.l2_normalize(self.rfeat_key[-1], dim=-1)
        # rfeat_norm = obs
        # rfeat_key_norm = self.rfeat_key[-1]
        self.rfeat_key[-1] = self.rfeat_key[-1] + pert_obs.mean(dim=0)


        sim = obs_key_norm * obs_norm # B, top_k, C
        tsim = torch.sum(sim) / obs.shape[0] # Scalar
        sim_loss['obs_sim'] = tsim

        sim2 = feat_key_norm * feat_norm # B, top_k, C
        tsim2 = torch.sum(sim2) / actor_features.shape[0] # Scalar
        sim_loss['feat_sim'] = tsim2

        # sim3 = rfeat_key_norm * rfeat_norm # B, top_k, C
        # tsim3 = torch.sum(sim3) / actor_features.shape[0] # Scalar
        # sim_loss['rfeat_sim'] = tsim3

        
        return sim_loss



    def max_consecutive_zeros(self, input_tensor):
        # Pad the tensor with ones on both sides
        padded = torch.nn.functional.pad(torch.unsqueeze(input_tensor, 0), (1, 1), value=1)
        
        # Compute differences along the columns
        diffs = torch.diff(padded, dim=1)
        
        # Find where zero sequences start (-1) and end (1)
        start_mask = (diffs == -1)
        end_mask = (diffs == 1)
        
        # Get row and column indices of starts and ends
        start_rows, start_cols = torch.where(start_mask)
        end_rows, end_cols = torch.where(end_mask)
        
        # Compute run lengths
        run_lengths = end_cols - start_cols
        
        # Initialize max_zeros and update manually
        max_zeros = torch.zeros(input_tensor.shape[0], dtype=torch.long)
        for row, length in zip(start_rows, run_lengths):
            max_zeros[row] = max(max_zeros[row], length)
        
        return max_zeros[0]