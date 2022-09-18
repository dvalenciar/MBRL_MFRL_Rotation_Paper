import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(Critic, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=num_actions)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = self.h_linear_4(x)                  # No activation function here
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.bn1 = nn.BatchNorm1d(hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.bn1(self.h_linear_3(x)))
        x = torch.tanh(self.h_linear_4(x))
        return x



# -------------------Networks for Model Learning -----------------------------#
class ModelNet_probabilistic_transition(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNet_probabilistic_transition, self).__init__()

        self.number_mixture_gaussians = 3

        self.initial_shared_layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size[0]),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
        )

        self.phi_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 15 * self.number_mixture_gaussians),
            nn.Softmax()
        )

        self.mean_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 15 * self.number_mixture_gaussians)
        )

        self.std_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 15 * self.number_mixture_gaussians),
            nn.Softplus()
        )

        '''
        nn.init.xavier_normal_(self.initial_shared_layer[1].weight.data)
        nn.init.ones_(self.initial_shared_layer[1].bias.data)
        nn.init.xavier_normal_(self.initial_shared_layer[4].weight.data)
        nn.init.ones_(self.initial_shared_layer[4].bias.data)

        nn.init.xavier_normal_(self.mean_layer[1].weight.data)
        nn.init.ones_(self.mean_layer[1].bias.data)
        nn.init.xavier_normal_(self.mean_layer[3].weight.data)
        nn.init.ones_(self.mean_layer[3].bias.data)

        nn.init.xavier_normal_(self.std_layer[1].weight.data)
        nn.init.ones_(self.std_layer[1].bias.data)
        nn.init.xavier_normal_(self.std_layer[3].weight.data)
        nn.init.ones_(self.std_layer[3].bias.data)

        nn.init.xavier_normal_(self.phi_layer[1].weight.data)
        nn.init.ones_(self.phi_layer[1].bias.data)
        nn.init.xavier_normal_(self.phi_layer[3].weight.data)
        nn.init.ones_(self.phi_layer[3].bias.data)
        '''

    def forward(self, state, action):

        x   = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x   = self.initial_shared_layer(x)

        u   = self.mean_layer(x)
        std = torch.clamp(self.std_layer(x), min=0.001)
        phi = self.phi_layer(x)

        u   = torch.reshape(u,   (-1, 15, self.number_mixture_gaussians))
        std = torch.reshape(std, (-1, 15, self.number_mixture_gaussians))
        phi = torch.reshape(phi, (-1, 15, self.number_mixture_gaussians))

        mix        = torch.distributions.Categorical(phi)
        norm_distr = torch.distributions.Normal(u, std)

        #comp = torch.distributions.Independent(norm_distr, 1)
        gmm = torch.distributions.MixtureSameFamily(mix, norm_distr)

        return gmm
