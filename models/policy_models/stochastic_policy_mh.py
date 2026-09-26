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

class StochasticPolicyMH(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicyMH, self).__init__()
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
            ).to(self.device)])


        self.act = torch.nn.ModuleList([ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        ).to(self.device)])


        # self.obs_key = [nn.Parameter(torch.zeros(self.args["max_obs_length"])).to(self.device)]
        # self.feat_key = [nn.Parameter(torch.zeros(self.hidden_sizes[-1])).to(self.device)]

        self.obs_key = nn.ParameterList([nn.Parameter(torch.randn(self.args["max_obs_length"])).to(self.device)])
        self.feat_key = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_sizes[-1])).to(self.device)])

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
        # self.act.append(copy.deepcopy(self.act[-1]))

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn.append(
                RNNLayer(
                    self.hidden_sizes[-1],
                    self.hidden_sizes[-1],
                    self.recurrent_n,
                    self.initialization_method,
                ).to(self.device)
            )

        self.act.append(
            ACTLayer(
                # action_space,
                # self.args["max_act_length"],
                self.action_space,
                self.hidden_sizes[-1],
                self.initialization_method,
                self.gain,
                self.args,
                # args
            ).to(self.device)
        )

        for i in range(len(self.obs_key)):
            self.obs_key[i].requires_grad = False
            self.obs_key[i].requires_grad = False
            for param in self.rnn[i].parameters():
                param.requires_grad = False
            for param in self.act[i].parameters():
                param.requires_grad = False

        # self.obs_key.append(nn.Parameter(torch.zeros(self.args["max_obs_length"])).to(self.device))
        # self.feat_key.append(nn.Parameter(torch.zeros(self.hidden_sizes[-1])).to(self.device))
        self.obs_key.append(nn.Parameter(torch.randn(self.args["max_obs_length"])).to(self.device))
        self.feat_key.append(nn.Parameter(torch.randn(self.hidden_sizes[-1])).to(self.device))

        

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


        actor_features = self.base(obs)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # if self.training or len(self.act)==1:  
        if  len(self.act)==1 or (not self.eval_mode):  
            # print("FW TR last head selection, head len = ", len(self.act))
            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                actor_features, rnn_states = self.rnn[-1](actor_features, rnn_states, masks)
            
            actions, action_log_probs = self.act[-1](
                actor_features, available_actions, deterministic
            )
        else:
            # print("randomized head selection")
            # print("FW TS randomized head selection,  head len = ", len(self.act))
            # idx = random.randint(0, len(self.act)-1)
            idx = self.predict_head_idx(obs, actor_features)
            # print("predicted head: ", idx)
            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                actor_features, rnn_states = self.rnn[idx](actor_features, rnn_states, masks)
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

        actor_features = self.base(obs)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # if self.training or len(self.act)==1:  
        if  len(self.act)==1 or (not self.eval_mode):  
            # print("ACEV TR last head selection, head len = ", len(self.act))
            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                actor_features, rnn_states = self.rnn[-1](actor_features, rnn_states, masks)
            
            action_log_probs, dist_entropy, action_distribution = self.act[-1].evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self.use_policy_active_masks else None,
            )

        else:
            # print("EVA TS randomized head selection,  head len = ", len(self.act))
            # idx = random.randint(0, len(self.act)-1)
            idx = self.predict_head_idx(obs, actor_features)
            # print("predicted head: ", idx)
            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                actor_features, rnn_states = self.rnn[idx](actor_features, rnn_states, masks)

            action_log_probs, dist_entropy, action_distribution = self.act[idx].evaluate_actions(
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

            head_sim.append(tsim * tsim2)
            # head_sim.append(tsim2)
            # head_sim.append(torch.norm(tsim) * torch.norm(tsim2))

        # print("head sim: ", torch.stack(head_sim).item(), end=" ")
        return torch.argmax(torch.stack(head_sim))



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

        actor_features = self.base(obs)

        extra_loss = {}
        if self.training:
            obs_norm = self.l2_normalize(obs, dim=-1) 
            obs_key_norm = self.l2_normalize(self.obs_key[-1], dim=-1) 
            feat_norm = self.l2_normalize(actor_features, dim=-1) 
            feat_key_norm = self.l2_normalize(self.feat_key[-1], dim=-1)

            # obs_norm = obs
            # obs_key_norm = self.obs_key[-1] 
            # feat_norm = actor_features
            # feat_key_norm = self.feat_key[-1]


            sim = obs_key_norm * obs_norm # B, top_k, C
            tsim = torch.sum(sim) / obs.shape[0] # Scalar
            extra_loss['obs_sim'] = tsim

            sim2 = feat_key_norm * feat_norm # B, top_k, C
            tsim2 = torch.sum(sim2) / actor_features.shape[0] # Scalar
            extra_loss['feat_sim'] = tsim2

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # print("TRACEV last head selection, head len = ", len(self.act))
        # if self.training or len(self.act)==1:  
        if  len(self.act)==1 or (not self.eval_mode):    
            # print("last head selection, head len = ", len(self.act))
            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                actor_features, rnn_states = self.rnn[-1](actor_features, rnn_states, masks)
            
            action_log_probs, dist_entropy, action_distribution = self.act[-1].evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self.use_policy_active_masks else None,
            )
            
        else:
            # print("EVA randomized head selection,  head len = ", len(self.act))
            # idx = random.randint(0, len(self.act)-1)
            idx = self.predict_head_idx(obs, actor_features)
            # print("predicted head: ", idx)
            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                actor_features, rnn_states = self.rnn[idx](actor_features, rnn_states, masks)

            action_log_probs, dist_entropy, action_distribution = self.act[idx].evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self.use_policy_active_masks else None,
            )




        return action_log_probs, dist_entropy, action_distribution, extra_loss

