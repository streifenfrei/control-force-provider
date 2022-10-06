import os.path
import random
import re
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque, defaultdict
from abc import ABC, abstractmethod
from functools import reduce
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'velocity', 'action', 'next_state', 'reward'))
epsilon = 1e-9


class StateAugmenter:
    class _StatePartID(str, Enum):
        robot_position = "ree"
        robot_velocity = "rve"
        robot_rotation = "rro"
        robot_rcm = "rpp"
        obstacle_position = "oee"
        obstacle_velocity = "ove"
        obstacle_rotation = "oro"
        obstacle_rcm = "opp"
        goal = "gol"
        time = "tim"

    _pattern_regex = "([a-z]{3})\\((([a-z][0-9]+)*)\\)"
    _arg_regex = "[a-z][0-9]+"

    def __init__(self, state_pattern, num_obstacles, ob_sigma, ob_max_noise):
        self.mapping = {}
        ids = re.findall(self._pattern_regex, state_pattern)
        index = 0
        for id in ids:
            args = id[1]
            id = id[0]
            history_length = 1
            for arg in re.findall(self._arg_regex, args):
                if arg[0] == "h":
                    history_length = int(arg[1:])
            if id == StateAugmenter._StatePartID.time:
                length = 1
            elif id in [StateAugmenter._StatePartID.robot_rotation, StateAugmenter._StatePartID.obstacle_rotation]:
                length = 4
            else:
                length = 3
            length *= num_obstacles if id in [StateAugmenter._StatePartID.obstacle_position,
                                              StateAugmenter._StatePartID.obstacle_velocity,
                                              StateAugmenter._StatePartID.obstacle_rotation,
                                              StateAugmenter._StatePartID.obstacle_rcm] else 1
            length *= history_length
            self.mapping[id] = (index, length)
            index += length
        self.num_obstacles = num_obstacles
        self.rng = np.random.default_rng()
        self.ob_sigma = ob_sigma
        self.ob_max_noise = ob_max_noise

    def __call__(self, state):
        index, length = self.mapping[StateAugmenter._StatePartID.obstacle_position]
        state[:, index:index + length] += np.clip(self.rng.multivariate_normal(np.zeros(length), np.identity(length) * self.ob_sigma, len(state)), -self.ob_max_noise, self.ob_max_noise)
        return state


class RewardFunction:

    def __init__(self, fmax, interval_duration, dc, mc, max_penalty, dg, rg):
        self.fmax = float(fmax)
        self.dc = float(dc)
        self.mc = float(mc)
        self.max_penalty = max_penalty
        self.dg = float(dg)
        self.rg = float(rg)
        self.interval_duration = float(interval_duration)

    def __call__(self, state_dict, last_state_dict):
        goal = np.array(state_dict["goal"])
        robot_position = np.array(state_dict["robot_position"])
        last_robot_position = np.array(last_state_dict["robot_position"])
        distance_vectors = (np.array(state_dict["points_on_l2"][x:x + 3]) - np.array(state_dict["points_on_l1"][x:x + 3]) for x in range(0, len(state_dict["points_on_l1"]), 3))
        distance_to_goal = np.linalg.norm(goal - robot_position)
        motion_reward = (np.linalg.norm(goal - last_robot_position) - distance_to_goal) / (self.fmax * self.interval_duration)
        collision_penalty = reduce(lambda x, y: x + y, ((self.dc / (np.linalg.norm(o) + 1e-10)) ** self.mc for o in distance_vectors))
        collision_penalty = 0 if any(np.any(np.isnan(x)) for x in distance_vectors) else - np.minimum(collision_penalty, self.max_penalty)
        goal_reward = 0 if distance_to_goal > self.dg else self.rg
        total_reward = motion_reward + collision_penalty + goal_reward
        return total_reward, motion_reward, collision_penalty, goal_reward


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        invalid = any(np.any(np.isnan(arg)) or np.any(np.isinf(arg)) for arg in args)
        if not invalid:
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
        mu = self.mu_dense(x)
        mu_norm = torch.linalg.norm(mu, dim=-1, keepdim=True)
        mu = (mu / mu_norm) * torch.tanh(mu_norm) * self.max_force
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
    class Accumulator:
        state = 0
        count = 0

        def update_state(self, value):
            self.state += value
            self.count += 1

        def get_value(self):
            return self.state / self.count

        def reset(self):
            self.state = 0
            self.count = 0

    def __init__(self,
                 state_dim,
                 action_dim,
                 discount_factor,
                 batch_size,
                 updates_per_step,
                 max_force,
                 reward_function,
                 state_augmenter,
                 output_directory,
                 exploration_angle_sigma,
                 exploration_magnitude_sigma,
                 exploration_decay,
                 exploration_duration,
                 dot_loss_factor,
                 dot_loss_decay,
                 **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.max_force = float(max_force)
        self.output_dir = output_directory
        self.save_file = os.path.join(self.output_dir, "save.pt")
        self.summary_writer = SummaryWriter(os.path.join(output_directory, "logs"))
        self.episode_accumulators = defaultdict(RLContext.Accumulator)
        self.reward_function = reward_function
        self.state_augmenter = state_augmenter
        self.last_state_dict = None
        self.action = None
        self.epoch = 0
        self.episode = 0
        self.episode_start = 0
        self.goal = None
        self.rng = np.random.default_rng()
        self.exploration_angle_sigma = exploration_angle_sigma
        self.exploration_rot_axis = None
        self.exploration_angle = 0
        self.exploration_magnitude_sigma = exploration_magnitude_sigma
        self.exploration_magnitude = 0
        self.exploration_decay = exploration_decay
        self.exploration_duration = exploration_duration
        self.exploration_index = 0
        self.dot_loss_factor = dot_loss_factor
        self.dot_loss_decay = dot_loss_decay

    def __del__(self):
        self.summary_writer.flush()

    @abstractmethod
    def _get_state_dict(self):
        return

    @abstractmethod
    def _load_impl(self, state_dict):
        return

    @abstractmethod
    def _update_impl(self, state_dict, reward):
        return

    def save(self):
        state_dict = self._get_state_dict()
        state_dict = {**state_dict,
                      "epoch": self.epoch,
                      "episode": self.episode - 1,
                      "exploration_angle_sigma": self.exploration_angle_sigma,
                      "exploration_magnitude_sigma": self.exploration_magnitude_sigma}
        torch.save(state_dict, self.save_file)

    def load(self):
        if os.path.exists(self.save_file):
            state_dict = torch.load(self.save_file)
            self.epoch = state_dict["epoch"]
            self.episode = state_dict["episode"]
            self.exploration_angle_sigma = state_dict["exploration_angle_sigma"]
            self.exploration_magnitude_sigma = state_dict["exploration_magnitude_sigma"]
            self._load_impl(state_dict)

    def update(self, state_dict):
        for key in state_dict:
            state_dict[key] = np.array(state_dict[key])
        goal = state_dict["goal"]
        if (goal != self.goal).any():
            if self.goal is not None:
                for key in self.episode_accumulators:
                    self.summary_writer.add_scalar(key, self.episode_accumulators[key].get_value(), self.episode)
                    self.episode_accumulators[key].reset()
                self.summary_writer.add_scalar("steps_per_episode", self.epoch - self.episode_start, self.episode)
                self.episode += 1
            self.goal = goal
            self.episode_start = self.epoch
            self.last_state_dict = None
        reward = None
        if self.last_state_dict is not None:
            # get reward
            reward, motion_reward, collision_penalty, goal_reward = self.reward_function(state_dict, self.last_state_dict)
            self.summary_writer.add_scalar("reward/per_epoch/total", reward, self.epoch)
            self.episode_accumulators["reward/per_episode/total"].update_state(reward)
            self.summary_writer.add_scalar("reward/per_epoch/motion", motion_reward, self.epoch)
            self.episode_accumulators["reward/per_episode/motion"].update_state(motion_reward)
            self.summary_writer.add_scalar("reward/per_epoch/collision", collision_penalty, self.epoch)
            self.episode_accumulators["reward/per_episode/collision"].update_state(collision_penalty)
            self.summary_writer.add_scalar("reward/per_epoch/goal/", goal_reward, self.epoch)
            self.episode_accumulators["reward/per_episode/goal"].update_state(goal)
        self._update_impl(state_dict, reward)
        self.last_state_dict = state_dict
        # exploration
        if self.exploration_index == 0:
            self.exploration_rot_axis = self.rng.multivariate_normal(np.zeros(3), np.identity(3))
            self.exploration_angle = np.deg2rad(self.rng.normal(scale=self.exploration_angle_sigma))
            self.exploration_magnitude = self.rng.normal(scale=self.exploration_magnitude_sigma)
        self.exploration_index = (self.exploration_index + 1) % self.exploration_duration
        # rotate the action vector a bit
        self.exploration_rot_axis[2] = - (self.exploration_rot_axis[0] * self.action[0] + self.exploration_rot_axis[1] * self.action[1]) / self.action[2]  # make it perpendicular to the action vector
        self.exploration_rot_axis /= np.linalg.norm(self.exploration_rot_axis)
        self.exploration_rot_axis *= self.exploration_angle
        self.exploration_angle_sigma *= self.exploration_decay
        self.action = Rotation.from_rotvec(self.exploration_rot_axis).apply(self.action)
        # change magnitude
        magnitude = np.linalg.norm(self.action)
        clipped_magnitude = np.clip(magnitude + self.exploration_magnitude, 0., self.max_force)
        self.summary_writer.add_scalar("magnitude", clipped_magnitude, self.epoch)
        self.action = self.action / magnitude * clipped_magnitude
        self.exploration_magnitude_sigma *= self.exploration_decay
        self.summary_writer.add_scalar("exploration/angle_sigma", self.exploration_angle_sigma, self.epoch)
        self.summary_writer.add_scalar("exploration/magnitude_sigma", self.exploration_magnitude_sigma, self.epoch)
        return self.action


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
        self.ts_model = os.path.join(self.output_dir, "dqn_ts.pt")
        self.total_loss_accumulator = RLContext.Accumulator()
        self.rl_loss_accumulator = RLContext.Accumulator()
        self.dot_loss_accumulator = RLContext.Accumulator()

    def _get_state_dict(self):
        return {"model_state_dict": self.dqn_policy.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "dot_loss_factor": self.dot_loss_factor}
        # torch.jit.script(self.dqn_policy).save(self.ts_model)

    def _load_impl(self, state_dict):
        self.dqn_policy.load_state_dict(state_dict["model_state_dict"])
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.dot_loss_factor = state_dict["dot_loss_factor"]

    def _update_impl(self, state_dict, reward):
        if reward is not None:
            self.replay_buffer.push(self.last_state_dict["state"], state_dict["robot_velocity"][:3], self.action, state_dict["state"], reward)
        if len(self.replay_buffer) >= self.batch_size:
            for i in range(self.updates_per_step):
                batch = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))
                state_batch = torch.tensor(self.state_augmenter(np.stack(batch.state)), dtype=torch.float32).to(device)
                velocity_batch = torch.tensor(np.stack(batch.velocity), dtype=torch.float32).to(device)
                action_batch = torch.tensor(np.stack(batch.action), dtype=torch.float32).to(device)
                reward_batch = torch.tensor(np.stack(batch.reward), dtype=torch.float32).unsqueeze(-1).to(device)
                mu, q, _ = self.dqn_policy(state_batch, action_batch)
                with torch.no_grad():
                    v_target = self.dqn_target(state_batch)[2]
                target = reward_batch + self.discount_factor * v_target
                rl_loss = (1 - self.dot_loss_factor) * nn.MSELoss()(q, target)
                dot_loss = - self.dot_loss_factor * torch.mean(
                    torch.bmm((mu / (torch.norm(mu, dim=1).unsqueeze(1) + epsilon)).unsqueeze(1), (velocity_batch / (torch.norm(velocity_batch, dim=1).unsqueeze(1) + epsilon)).unsqueeze(2)).squeeze(2))
                loss = rl_loss + dot_loss
                self.rl_loss_accumulator.update_state(rl_loss.detach().numpy())
                self.dot_loss_accumulator.update_state(dot_loss.detach().numpy())
                self.total_loss_accumulator.update_state(loss.detach().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.dqn_policy.parameters(), 1)
                self.optimizer.step()
            self.summary_writer.add_scalar("loss/rl", self.rl_loss_accumulator.get_value(), self.epoch)
            self.summary_writer.add_scalar("loss/dot", self.dot_loss_accumulator.get_value(), self.epoch)
            self.summary_writer.add_scalar("loss/total", self.total_loss_accumulator.get_value(), self.epoch)
            self.rl_loss_accumulator.reset()
            self.dot_loss_accumulator.reset()
            self.total_loss_accumulator.reset()
            self.summary_writer.add_scalar("dot_loss_factor", self.dot_loss_factor, self.epoch)
            self.dot_loss_factor *= self.dot_loss_decay
            self.epoch += 1
        if self.epoch % self.target_network_update_rate == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_policy.eval()
        with torch.no_grad():
            self.action = self.dqn_policy(torch.tensor(state_dict["state"], dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
        self.dqn_policy.train()


context_mapping = {"dqn": DQNContext}
