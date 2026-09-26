import torch.nn as nn
import torch
from harl.utils.models_tools import init, get_active_func, get_init_method

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    """A MLP base module."""

    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        self.obs_dim = obs_shape[0]
        # self.input_size = 256
        # self.input_size = 2048
        # if args["max_obs_length"] is not None:
        if "max_obs_length" in args.keys():
            self.input_size = args["max_obs_length"]
        else:
            self.input_size = 384
   
    
        if self.use_feature_normalization:
            # self.feature_norm = nn.LayerNorm(obs_dim)
            self.feature_norm = nn.LayerNorm(self.input_size)

        self.mlp = MLPLayer(
            # obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
            self.input_size, self.hidden_sizes, self.initialization_method, self.activation_func
        )

    def forward(self, x):
        x = self.pad_input(x)

        if self.use_feature_normalization:
            x = self.feature_norm(x)
    
        x = self.mlp(x)

        return x

    def pad_input(self, x):
        # print(x.shape)
        pad = torch.zeros(x.shape[0], self.input_size-x.shape[1]).to(x.device)
        x = torch.cat((x,pad),dim=-1)
        return x 