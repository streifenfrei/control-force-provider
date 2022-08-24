import os.path
import re
import random
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
from abc import ABC, abstractmethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

pattern_regex = "([a-z]{3})\\((([a-z][0-9]+)*)\\)"
arg_regex = "[a-z][0-9]+"


def parse_state_pattern(state_pattern, num_obstacles):
    ids = re.findall(pattern_regex, state_pattern)
    structure = []
    index = 0
    for id in ids:
        args = id[1]
        id = id[0]
        history_length = 1
        for arg in re.findall(arg_regex, args):
            if arg[0] == "h":
                history_length = int(arg[1:])
        length = 1 if id == "tim" else 3
        length *= num_obstacles if id == "oee" or id == "opp" else 1
        length *= history_length
        structure.append((id, index, length))
        index += length
    return structure


class RewardFunction:

    def __init__(self, state_structure):
        self.state_structure = state_structure

    def __call__(self, state):
        return torch.tensor([0])


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        valid = True
        for arg in args:
            if torch.max(torch.isnan(arg)) or torch.max(torch.isinf(arg)):
                valid = False
                break
        if valid:
            self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, layer_size, max_force):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        self.max_force = float(max_force)
        self.dense1 = nn.Linear(state_dim, layer_size)
        self.bnorm1 = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.bnorm2 = nn.BatchNorm1d(layer_size)
        self.dense3 = nn.Linear(layer_size, layer_size)
        self.bnorm3 = nn.BatchNorm1d(layer_size)
        self.mu_dense = nn.Linear(layer_size, action_dim)
        self.v_dense = nn.Linear(layer_size, 1)
        self.l_dense = nn.Linear(layer_size, int(action_dim * (action_dim + 1) / 2))

    def forward(self, state, action=None):
        x = torch.relu(self.dense1(state))
        x = self.bnorm1(x)
        x = torch.relu(self.dense2(x))
        x = self.bnorm2(x)
        x = torch.relu(self.dense3(x))
        x = self.bnorm3(x)
        mu = self.max_force * torch.tanh(self.mu_dense(x))
        l_entries = torch.tanh(self.l_dense(x))
        v = self.v_dense(x)

        l = torch.zeros((state.shape[0], self.action_dim, self.action_dim)).to(device)
        indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        l[:, indices[0], indices[1]] = l_entries
        l.diagonal(dim1=1, dim2=2).exp_()
        p = l * l.transpose(2, 1)

        q = None
        if action is not None:
            action_diff = (action - mu).unsqueeze(-1)
            a = (-0.5 * torch.matmul(torch.matmul(action_diff.transpose(1, 2), p), action_diff)).squeeze(-1)
            q = a + v

        return mu, q, v


class RLContext(ABC):
    def __init__(self, state_dim, action_dim, reward_function, discount_factor, batch_size, updates_per_step, max_force, output_directory, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.max_force = max_force
        self.output_dir = output_directory
        self.summary_writer = SummaryWriter(os.path.join(output_directory, "logs"))
        self.reward_function = reward_function
        self.last_state = None
        self.action = None
        self.epoch = 0
        self.goal = None

    def __del__(self):
        self.summary_writer.flush()

    @abstractmethod
    def save(self): return

    @abstractmethod
    def load(self): return

    @abstractmethod
    def update(self, state, goal): return


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
        self.save_file = os.path.join(self.output_dir, "dqn.pt")
        self.ts_model = os.path.join(self.output_dir, "dqn_ts.pt")

    def save(self):
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.dqn_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, self.save_file)
        # torch.jit.script(self.dqn_policy).save(self.ts_model)

    def load(self):
        if os.path.exists(self.save_file):
            state_dict = torch.load(self.save_file)
            self.epoch = state_dict["epoch"]
            self.dqn_policy.load_state_dict(state_dict["model_state_dict"])
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def update(self, state, goal):
        reward = self.reward_function(state)
        self.summary_writer.add_scalar("reward/train", reward, self.epoch)
        state = torch.tensor(state)
        if goal != self.goal:
            self.summary_writer.add_text("episodes/train", f"New goal: {goal}", self.epoch)
            self.goal = goal
        if self.last_state is not None:
            self.replay_buffer.push(self.last_state, self.action, state, reward)
        self.last_state = state
        if len(self.replay_buffer) >= self.batch_size:
            for i in range(self.updates_per_step):
                batch = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))
                state_batch = torch.stack(batch.state).to(device)
                action_batch = torch.stack(batch.action).to(device)
                reward_batch = torch.stack(batch.reward).to(device)
                q = self.dqn_policy(state_batch, action_batch)[1]
                with torch.no_grad():
                    v_target = self.dqn_target(state_batch)[2]
                target = reward_batch + self.discount_factor * v_target
                loss = nn.MSELoss()(q, target)
                self.summary_writer.add_scalar("loss/train", loss, self.epoch)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.dqn_policy.parameters(), 1)
                self.optimizer.step()
        self.epoch += 1
        if self.epoch % self.target_network_update_rate == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_policy.eval()
        with torch.no_grad():
            self.action = self.dqn_policy(state.unsqueeze(0))[0].squeeze(0)
        self.dqn_policy.train()
        return self.action.tolist()


context_mapping = {"dqn": DQNContext}
