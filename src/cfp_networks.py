import os.path
import random
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class RewardFunction:
    def __call__(self, state):
        return 0


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, layer_size, max_force):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        self.max_force = max_force

        self.dense1 = nn.Linear(state_dim, layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.mu_dense = nn.Linear(layer_size, action_dim)
        self.v_dense = nn.Linear(layer_size, 1)
        self.l_dense = nn.Linear(layer_size, int(action_dim * (action_dim + 1) / 2))

    def forward(self, state, action=None):
        action = action.unsqueeze(1)
        x = torch.relu(self.dense1(state))
        x = torch.relu(self.dense2(x))
        mu = self.max_force * torch.tanh(self.l_dense(x)).unsqueeze(1)
        v = self.v_dense(x)

        l = torch.zeros((state.size[0], self.action_dim, self.action_dim))
        indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        p = l * l.transpose(2, 1)

        q = None
        if action is not None:
            a = (-0.5 * torch.matmul(torch.matmul((action - mu).transpose(2, 1), p), (action - mu))).squeeze(-1)
            q = a + v

        return mu, q, v


class RLContext:
    def __init__(self, state_dim, action_dim, discount_factor, batch_size, updates_per_step, max_force, output_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.max_force = max_force
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "cfp_rl.log")
        self.reward_function = RewardFunction()
        self.last_state = None
        self.action = None
        self.epoch = 0


class DQNContext(RLContext):
    def __init__(self, layer_size, replay_buffer_size, target_network_update_rate, **kwargs):
        super().__init__(**kwargs)
        self.dqn_policy = DQN(self.state_dim, self.action_dim, layer_size, self.max_force).to(device)
        self.dqn_target = DQN(self.state_dim, self.action_dim, layer_size, self.max_force).to(device)
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_target.eval()
        self.optimizer = torch.optim.Adam(self.dqn_policy.parameters())
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.target_network_update_rate = target_network_update_rate

    def update(self, state):
        self.replay_buffer.push(self.last_state, self.action, state, self.reward_function(state))
        self.last_state = state
        if len(self.replay_buffer) >= self.batch_size:
            for i in range(self.updates_per_step):
                batch = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))
                state_batch = torch.cat(batch.state).to(device)
                action_batch = torch.cat(batch.action).to(device)
                reward_batch = torch.cat(batch.reward).to(device)
                q = self.dqn_policy(state_batch, action_batch)[1]
                with torch.no_grad():
                    v_target = self.dqn_target(state_batch)[2]
                target = reward_batch + self.discount_factor * v_target
                loss = nn.MSELoss()(q, target)
                self.optimizer.zero_grad()
                loss.backwards()
                clip_grad_norm_(self.dqn_policy.parameters(), 1)
                self.optimizer.step()

        self.epoch += 1
        if self.epoch % self.target_network_update_rate == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        with torch.no_grad():
            self.action = self.dqn_policy(state)[0]
        return self.action
