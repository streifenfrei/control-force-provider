import os
import re
import math
import torch
import numpy as np
import rospy
from enum import Enum
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, defaultdict, deque
from threading import Thread, Lock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-9
BB_REPULSION_DISTANCE = 1e-2

Transition = namedtuple('Transition', ('state', 'velocity', 'action', 'next_state', 'reward', 'is_terminal'))


class StatePartID(str, Enum):
    robot_position = "ree"
    robot_velocity = "rve"
    robot_rotation = "rro"
    robot_rcm = "rpp"
    obstacle_position = "oee"
    obstacle_direction = "odi"
    obstacle_velocity = "ove"
    obstacle_rotation = "oro"
    obstacle_rcm = "opp"
    goal = "gol"
    time = "tim"


_pattern_regex = "([a-z]{3})\\((([a-z][0-9]+)*)\\)"
_arg_regex = "[a-z][0-9]+"


def create_state_mapping(state_pattern, num_obstacles):
    mapping = {}
    ids = re.findall(_pattern_regex, state_pattern)
    index = 0
    for id in ids:
        args = id[1]
        id = id[0]
        history_length = 1
        for arg in re.findall(_arg_regex, args):
            if arg[0] == "h":
                history_length = int(arg[1:])
        if id == StatePartID.time:
            length = 1
        elif id in [StatePartID.robot_rotation, StatePartID.obstacle_rotation]:
            length = 4
        else:
            length = 3
        length *= num_obstacles if id in [StatePartID.obstacle_position,
                                          StatePartID.obstacle_velocity,
                                          StatePartID.obstacle_rotation,
                                          StatePartID.obstacle_direction,
                                          StatePartID.obstacle_rcm] else 1
        length *= history_length
        mapping[id] = (index, length)
        index += length
    return mapping


class StateAugmenter:

    def __init__(self, mapping, ob_sigma, ob_max_noise):
        self.mapping = mapping
        self.ob_sigma = ob_sigma
        self.ob_max_noise = ob_max_noise
        self.noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * self.ob_sigma)

    def __call__(self, state):
        index, length = self.mapping[StatePartID.obstacle_position]
        stack = []
        for i in range(0, length, 3):
            stack.append(self.noise_dist.sample([state.size(0)]).to(state.device))
        state[:, index:index + length] += torch.clip(torch.cat(stack, 1), -self.ob_max_noise, self.ob_max_noise)
        return state


class RewardFunction:
    def __init__(self, fmax, interval_duration, dc, mc, motion_penalty, min_timeout_penalty, max_timeout_penalty, max_penalty, dg, rg):
        self.fmax = torch.tensor(float(fmax), device=DEVICE)
        self.dc = torch.tensor(float(dc), device=DEVICE)
        self.mc = torch.tensor(float(mc), device=DEVICE)
        self.motion_penalty = motion_penalty
        self.min_timeout_penalty = min_timeout_penalty
        self.timeout_penalty_range = max_timeout_penalty - min_timeout_penalty
        self.max_penalty = torch.tensor(float(max_penalty), device=DEVICE)
        self.dg = torch.tensor(float(dg), device=DEVICE)
        self.rg = torch.tensor(float(rg), device=DEVICE)
        self.interval_duration = float(interval_duration)
        self.max_distance = None

    def __call__(self, state_dict, last_state_dict, mask):
        goal = state_dict["goal"]
        if not len(last_state_dict):
            out = torch.full([goal.size(0), 1], torch.nan, device=DEVICE)
            return out, out, out, out
        #robot_position = state_dict["robot_position"]
        # last_robot_position = last_state_dict["robot_position"]
        #collision_distances = (state_dict["collision_distances"][:, x:x + 1] for x in range(state_dict["collision_distances"].size(-1)))
        #distance_to_goal = torch.linalg.norm(goal - robot_position, dim=-1).unsqueeze(-1)
        # motion_reward = torch.where(mask, (torch.linalg.norm(goal - last_robot_position) - distance_to_goal) / (self.fmax * self.interval_duration), torch.nan)
        collision_penalty = torch.where(mask, 0, torch.nan)
        collision_penalty -= torch.where(state_dict["collided"], self.max_penalty, 0)
        motion_reward = torch.where(mask, torch.full_like(collision_penalty, self.motion_penalty), torch.nan)
        #for collision_distance in collision_distances:
        #    collision_penalty = -torch.where(collision_distance.isnan(), 0, collision_penalty + (self.dc / (collision_distance + EPSILON)) ** self.mc)
        #collision_penalty = torch.minimum(collision_penalty, self.max_penalty)
        goal_reward = torch.where(mask, torch.where(state_dict["reached_goal"], self.rg, torch.zeros_like(collision_penalty)), torch.nan)
        #if self.max_distance is None:
        #    self.max_distance = torch.linalg.norm(state_dict["workspace_bb_dims"])
        #timeout_penalty = self.min_timeout_penalty + ((distance_to_goal / self.max_distance) * self.timeout_penalty_range)
        #goal_reward = torch.where(state_dict["reached_goal"], goal_reward, 0)
        total_reward = motion_reward + collision_penalty + goal_reward  # + timeout_penalty
        return total_reward, motion_reward, collision_penalty, goal_reward


class ActionSpace:
    def __init__(self, grid_order, magnitude_order, max_force):
        assert grid_order > 0
        assert magnitude_order > 0
        angle_step = math.pi / 2 / grid_order
        theta_step_num = 2 * grid_order
        phi_step_num = 4 * grid_order
        self.magnitude_step = max_force / magnitude_order
        self.action_vectors = []
        self.action_vectors_normalized = []
        for i in range(theta_step_num + 1):
            theta = i * angle_step
            for j in range(phi_step_num):
                phi = j * angle_step
                action_vector = torch.tensor([math.sin(theta) * math.cos(phi),
                                              math.sin(theta) * math.sin(phi),
                                              math.cos(theta)], device=DEVICE).unsqueeze(0)
                action_vector = torch.where(action_vector.abs() < EPSILON, 0, action_vector)
                for k in range(1, magnitude_order + 1):
                    self.action_vectors_normalized.append(action_vector)
                    self.action_vectors.append(action_vector * k * self.magnitude_step)
                if i == 0 or i == theta_step_num:
                    break
        self.action_space_tensor = torch.cat(self.action_vectors)
        self.goal_vectors = [torch.zeros([1, 3], device=DEVICE) for _ in range(magnitude_order)]
        self.goal_vectors_index_start = 0

    def update_goal_vector(self, goal_vector):
        goal_vector = goal_vector.to(DEVICE)
        norm = torch.linalg.vector_norm(goal_vector, dim=-1, keepdim=True)
        goal_vector /= norm + EPSILON
        min_angle = torch.full([goal_vector.size(0), 1], torch.inf, device=DEVICE)
        self.goal_vectors_index_start = torch.empty_like(min_angle, dtype=torch.int64)
        for i in range(0, len(self.action_vectors), len(self.goal_vectors)):
            angle = torch.acos(torch.linalg.vecdot(goal_vector, self.action_vectors_normalized[i])).unsqueeze(-1)
            mask = angle < min_angle
            min_angle = torch.where(mask, angle, min_angle)
            self.goal_vectors_index_start = torch.where(mask, i, self.goal_vectors_index_start)
        for i in range(len(self.goal_vectors)):
            self.goal_vectors[i] = goal_vector * torch.minimum(norm, torch.tensor((i + 1) * self.magnitude_step))

    def get_action(self, index):
        index = index.to(DEVICE)
        index = index.expand([-1, 3])
        actions = torch.gather(self.action_space_tensor, 0, index)
        # for i in range(len(self.goal_vectors)):
        #    actions = torch.where(index == self.goal_vectors_index_start + i, self.goal_vectors[i], actions)
        return actions

    def __len__(self):
        return len(self.action_vectors)


class SmartStateDict():
    def __init__(self):
        self.data = {}
        self.access_counts = defaultdict(int)

        def factory():
            return deque(maxlen=10)
        self.last_access_counts = defaultdict(factory)
        self.approx_max_access_counts = defaultdict(int)
        self.locks = defaultdict(Lock)

    def _copy_sync(self, key, device):
        with self.locks[key]:
            self.data[key] = self.data[key].to(device)

    def _copy_async(self, key, device):
        Thread(target=SmartStateDict._copy_sync, args=[self, key, device]).start()

    def update_data(self, data):
        self.data = data
        for key in self.data:
            self.last_access_counts[key].append(self.access_counts[key])
            self.approx_max_access_counts[key] = int(sum(self.last_access_counts[key]) / len(self.last_access_counts[key]))
        self.access_counts = defaultdict(int)

    def load(self, device=None):
        for key in self.data:
            if device is None:
                device = "cpu" if self.approx_max_access_counts[key] == 0 else DEVICE
            self._copy_async(key, device)

    def __getitem__(self, key):
        with self.locks[key]:
            if self.data[key].device != DEVICE:
                self._copy_sync(key, DEVICE)
            self.access_counts[key] += 1
            if self.access_counts[key] >= self.approx_max_access_counts[key]:
                self._copy_async(key, "cpu")
            return self.data[key]

    def __len__(self):
        return len(self.data)


class RLContext(ABC):
    class Accumulator:
        def __init__(self, batch_size=1):
            self.state = torch.zeros(batch_size)
            self.count = torch.zeros(batch_size)
            self.default_mask = torch.ones_like(self.state, dtype=torch.bool)

        def update_state(self, value, mask=None):
            if mask is None:
                mask = self.default_mask
            mask = mask.to(DEVICE)
            value = value.to(DEVICE)
            state_dev = self.state.to(DEVICE)
            self.state = torch.where(mask, state_dev + value, state_dev).cpu()
            count_dev = self.count.to(DEVICE)
            self.count = torch.where(mask, count_dev + 1, count_dev).cpu()

        def get_value(self, mask=None):
            if mask is None:
                mask = self.default_mask
            value = torch.where(torch.logical_and(mask, self.count != 0), self.state / self.count, 0)
            return torch.masked_select(value, mask).cpu().numpy()

        def reset(self, mask=None):
            if mask is None:
                mask = self.default_mask
            mask = mask.to(DEVICE)
            self.state = torch.where(mask, 0, self.state.to(DEVICE)).cpu()
            self.count = torch.where(mask, 0, self.count.to(DEVICE)).cpu()

    class AccumulatorFactory:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def __call__(self):
            return RLContext.Accumulator(self.batch_size)

    def __init__(self,
                 state_dim,
                 action_dim,
                 discount_factor,
                 batch_size,
                 robot_batch,
                 max_force,
                 reward_function,
                 goal_reached_threshold_distance,
                 goal_distance,
                 state_augmenter,
                 output_directory,
                 save_rate,
                 interval_duration,
                 episode_timeout,
                 train,
                 evaluation_duration,
                 log=True,
                 **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.robot_batch = robot_batch
        self.max_force = float(max_force)
        self.max_distance = None
        self.output_dir = output_directory
        self.save_rate = save_rate
        self.summary_writer = SummaryWriter(os.path.join(output_directory, "logs"), max_queue=10000, flush_secs=10)
        self.log_interval = save_rate
        self.log_dict = {}
        self.reward_function = reward_function
        self.state_augmenter = state_augmenter
        self.state_dict = SmartStateDict()
        self.last_state_dict = SmartStateDict()
        self.action = None
        self.epoch = 0
        self.episode = torch.zeros((robot_batch, 1))
        self.max_episode_length = int(episode_timeout / (interval_duration * 1e-3))
        self.total_episode_count = 0
        self.last_episode_count = 0
        self.goal_reached_threshold_distance = goal_reached_threshold_distance
        self.goal_distance = goal_distance
        self.interval_duration = interval_duration
        self.stop_update = False
        self.state_batch = None
        self.velocity_batch = None
        self.action_batch = None
        self.reward_batch = None
        self.successes = []
        self.collisions = []
        self.episodes_lengths = []
        self.episodes_passed = []
        self.timeouts = []
        self.returns = []
        self.acc_reward = torch.zeros([robot_batch, 1])
        self.acc_lengths = torch.zeros([robot_batch, 1])
        self.log = log
        self.train = train
        self.evaluating = False
        self.evaluation_duration = evaluation_duration
        self.evaluation_epoch = 0
        self.metrics = {}
        self.her_reward = self.reward_function.rg + self.reward_function.motion_penalty
        self.her_noise_dist = self.noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * 1e-5)
        self.her_transition_buffer = []
        self.goal_state_index = self.state_augmenter.mapping[StatePartID.goal][0]

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

    @abstractmethod
    def _post_update(self, state_dict):
        return

    def save(self):
        state_dict = self._get_state_dict()
        state_dict = {**state_dict,
                      "epoch": self.epoch,
                      "episode": self.total_episode_count - self.robot_batch,
                      "goal_distance": self.goal_distance
                      }
        torch.save(state_dict, os.path.join(self.output_dir, f"{self.epoch}.pt"))

    def load(self):
        save_file = None
        max_epoch = 0
        for file in os.listdir(self.output_dir):
            full_path = os.path.join(self.output_dir, file)
            if os.path.isfile(full_path) and file.endswith(".pt"):
                epoch = int(file[:-3])
                if epoch >= max_epoch:
                    save_file = full_path
                    max_epoch = epoch
        if save_file is not None:
            state_dict = torch.load(save_file)
            self.epoch = state_dict["epoch"]
            self.goal_distance = state_dict["goal_distance"]
            self._load_impl(state_dict)

    @staticmethod
    def _copy_state_dict(state_dict):
        state_dict_copy = {}
        for key, value in state_dict.items():
            state_dict_copy[key] = value.clone()
        return state_dict_copy

    def create_her_transitions(self, state_dict, action, next_state_dict, reward, single_transition=False):
        if single_transition:
            mask = next_state_dict["collided"].logical_not()
            if mask.any():
                states = state_dict["state"][mask.expand([-1, self.state_dim])].reshape([-1, self.state_dim]).clone()
                actions = action[mask.expand([-1, action.size(-1)])].reshape([-1, action.size(-1)])
                velocities = state_dict["robot_velocity"][mask.expand([-1, 3])].reshape([-1, 3])
                next_states = next_state_dict["state"][mask.expand([-1, self.state_dim])].reshape([-1, self.state_dim]).clone()
                rewards = torch.full([action.size(0), 1], self.her_reward)
                goals = next_state_dict["robot_position"][mask.expand([-1, 3])].reshape([-1, 3])
                noise = self.her_noise_dist.sample([goals.size(0)])
                noise_magnitudes = torch.linalg.vector_norm(noise, dim=-1, keepdims=True)
                noise /= noise_magnitudes
                noise *= torch.minimum(noise_magnitudes, torch.tensor(self.goal_reached_threshold_distance))
                goals += noise
                states[:, self.goal_state_index:self.goal_state_index + 3] = goals
                next_states[:, self.goal_state_index:self.goal_state_index + 3] = goals
                return states, velocities, actions, next_states, rewards, torch.ones([states.size(0), 1])
            else:
                return None
        is_her_terminal = next_state_dict["is_timeout"]
        self.her_transition_buffer.append((state_dict["state"],
                                           state_dict["robot_velocity"],
                                           action,
                                           next_state_dict["state"],
                                           torch.where(is_her_terminal, next_state_dict["robot_position"], torch.full_like(next_state_dict["robot_position"], torch.nan)),
                                           next_state_dict["is_terminal"],
                                           is_her_terminal,
                                           torch.where(is_her_terminal, self.her_reward, reward)))
        if len(self.her_transition_buffer) == self.max_episode_length:
            transitions = list(zip(*self.her_transition_buffer))
            states = torch.stack(transitions[0]).to(DEVICE)
            velocities = torch.stack(transitions[1]).to(DEVICE)
            actions = torch.stack(transitions[2]).to(DEVICE)
            next_states = torch.stack(transitions[3]).to(DEVICE)
            goals = torch.stack(transitions[4]).to(DEVICE)
            no_terminals = torch.stack(transitions[5]).to(DEVICE).logical_not()
            her_terminals = torch.stack(transitions[6]).to(DEVICE)
            rewards = torch.stack(transitions[7]).to(DEVICE)
            is_valid = torch.zeros_like(no_terminals).to(DEVICE)
            for i in reversed(range(self.max_episode_length)):
                i_plus_one = min(i + 1, self.max_episode_length - 1)
                is_valid[i, :, :] = (is_valid[i_plus_one, :, :].logical_and(no_terminals[i, :, :])).logical_or(her_terminals[i, :, :])
                goals[i, :, :] = torch.where(goals[i, :, :].isnan(), goals[i_plus_one, :, :], goals[i, :, :])
            if not is_valid.any():
                return None
            noise = self.her_noise_dist.sample([goals.size(1)]).to(DEVICE)
            noise_magnitudes = torch.linalg.vector_norm(noise, dim=-1, keepdims=True)
            noise /= noise_magnitudes
            noise *= torch.minimum(noise_magnitudes, torch.tensor(self.goal_reached_threshold_distance))
            goals += noise.unsqueeze(0)
            states[:, :, self.goal_state_index:self.goal_state_index + 3] = goals
            next_states[:, :, self.goal_state_index:self.goal_state_index + 3] = goals
            states = torch.masked_select(states, is_valid).reshape([-1, states.size(-1)])
            velocities = torch.masked_select(velocities, is_valid).reshape([-1, 3])
            actions = torch.masked_select(actions, is_valid).reshape([-1, actions.size(-1)])
            next_states = torch.masked_select(next_states, is_valid).reshape([-1, states.size(-1)])
            rewards = torch.masked_select(rewards, is_valid).unsqueeze(-1)
            is_terminals = torch.masked_select(her_terminals, is_valid).unsqueeze(-1)
            self.her_transition_buffer = []
            return states, velocities, actions, next_states, rewards, is_terminals.float()
        else:
            return None

    def update(self, state_dict):
        self.state_dict.update_data(state_dict)
        state_dict = self.state_dict
        state_dict.load()
        self.last_state_dict.load()
        reward = None
        is_terminal = state_dict["is_terminal"]
        if self.train:
            if self.max_distance is None:
                self.max_distance = torch.linalg.norm(state_dict["workspace_bb_dims"])
            # get reward
            mask = self.last_state_dict["is_terminal"].logical_not() if len(self.last_state_dict) else None
            reward, motion_reward, collision_penalty, goal_reward = self.reward_function(state_dict, self.last_state_dict, mask)
            acc_reward_dev = self.acc_reward.to(DEVICE)
            acc_reward_dev = torch.where(reward.isnan(), acc_reward_dev, acc_reward_dev + reward)
            mean_return = torch.masked_select(acc_reward_dev, is_terminal).mean()
            if not mean_return.isnan():
                self.returns.append(mean_return.item())
            acc_reward_dev = torch.where(is_terminal, 0, acc_reward_dev)
            self.acc_reward = acc_reward_dev.cpu()
        # do the update
        self._update_impl(state_dict, reward)
        self._post_update(state_dict)
        self.last_state_dict.update_data(state_dict.data)
        self.last_state_dict.load("cpu")
        if self.evaluating:
            acc_lengths_dev = self.acc_lengths.to(DEVICE) + 1
            self.evaluation_epoch += 1
            reached_goal_dev = state_dict["reached_goal"]
            collided_dev = state_dict["collided"]
            is_timeout_dev = state_dict["is_timeout"]
            self.successes.append(reached_goal_dev.sum().item())
            self.collisions.append(collided_dev.sum().item())
            self.timeouts.append(is_timeout_dev.sum().item())
            self.episodes_passed.append(is_terminal.sum().item())
            self.episodes_lengths.append(acc_lengths_dev[is_terminal].sum().item())
            acc_lengths_dev = torch.where(is_terminal, 0, acc_lengths_dev)
            if self.evaluation_epoch != 0 and self.evaluation_epoch % self.evaluation_duration == 0:
                self.metrics = {}
                episodes_passed = sum(list(self.episodes_passed)[-self.log_interval:])
                self.metrics["success_ratio"] = sum(self.successes[-self.log_interval:]) / (episodes_passed + EPSILON)
                self.metrics["collision_ratio"] = sum(self.collisions[-self.log_interval:]) / (episodes_passed + EPSILON)
                self.metrics["timeout_ratio"] = sum(self.timeouts[-self.log_interval:]) / (episodes_passed + EPSILON)
                mean_return = sum(self.returns) / (len(self.returns) + EPSILON)
                self.summary_writer.add_scalar("return", mean_return, self.epoch)
                self.returns = []
                episodes_lengths = list(self.episodes_lengths)[-self.log_interval:]
                self.metrics["mean_episode_length"] = sum(episodes_lengths) / (episodes_passed + EPSILON)
                for key in self.metrics:
                    self.summary_writer.add_scalar(f"metrics/{key}", self.metrics[key], self.epoch)
                self.summary_writer.add_scalar("goal_distance", self.goal_distance, self.epoch)
                if self.log:
                    string = f"Epoch {self.epoch} | "
                    for key, value in self.log_dict.items():
                        mean = value / self.log_interval
                        self.summary_writer.add_scalar(key, mean, self.epoch)
                        string += f"{key}: {mean}\t "
                        self.log_dict[key] = 0
                    string += f"return: {mean_return}"
                    print("\r", end="")
                    rospy.loginfo(string)
                self.evaluation_epoch = 0
                self.evaluating = False
                self.train = True
                self.successes = []
                self.collisions = []
                self.episodes_lengths = []
                self.episodes_passed = []
                self.timeouts = []
                self.returns = []
                self.acc_reward = torch.zeros([self.robot_batch, 1])
                self.acc_lengths = torch.zeros([self.robot_batch, 1])
            else:
                self.acc_lengths = acc_lengths_dev.cpu()
        else:
            self.epoch += 1
            if self.epoch > 0:
                progress = self.epoch % self.log_interval
                if progress == 0:
                    self.evaluating = True
                    self.train = False
                    print("\rEvaluating...", end="")
                else:
                    print(f"\r{100 * progress / self.log_interval}%\t", end="")
                if self.epoch % self.save_rate == 0:
                    self.save()
        self.goal_distance = state_dict["goal_distance"].item()
        return self.action

    def warn(self, string):
        if self.log:
            rospy.logwarn(string)


class DiscreteRLContext(RLContext):
    def __init__(self, grid_order, magnitude_order, exploration_epsilon, exploration_sigma, exploration_decay, **kwargs):
        super().__init__(**kwargs)
        self.action_space = ActionSpace(grid_order, magnitude_order, self.max_force)
        self.action_vectors = torch.stack(self.action_space.action_vectors_normalized).permute(1, 2, 0)
        self.exploration_probs = None
        self.exploration_epsilon = exploration_epsilon
        self.exploration_decay = exploration_decay
        self.exploration_sigma = exploration_sigma
        self.exploration_gauss_factor = 1 / (self.exploration_sigma * math.sqrt(2 * math.pi))
        self.one_minus_exploration_decay = 1 - self.exploration_decay
        self.action_index = None
        self.max_distance = None
        self.best_success_ratio = 0
        self.good_success_ratio = 0.9

    @abstractmethod
    def _get_state_dict_(self):
        return

    @abstractmethod
    def _load_impl_(self, state_dict):
        return

    @abstractmethod
    def _update_impl(self, state_dict, reward):
        return

    def _get_state_dict(self):
        return {**self._get_state_dict_(),
                "exploration_epsilon": self.exploration_epsilon,
                "best_success_ratio": self.best_success_ratio}

    def _load_impl(self, state_dict):
        self.exploration_epsilon = state_dict["exploration_epsilon"]
        self.best_success_ratio = state_dict["best_success_ratio"]
        self._load_impl_(state_dict)

    def get_exploration_probs(self, action_probs, velocity):
        action_probs = action_probs.to(DEVICE)
        velocity = velocity.to(DEVICE)
        batch_size = action_probs.size(0)
        # exploration probs depend on the angle to the current velocity (normal distribution)
        velocities_normalized = velocity / (torch.linalg.vector_norm(velocity, dim=-1, keepdim=True) + EPSILON)
        angles = torch.matmul(velocities_normalized.view([batch_size, 1, 3]), self.action_vectors)
        angles = torch.acos(torch.clamp(angles, -1, 1)).squeeze()
        exploration_probs = self.exploration_gauss_factor * torch.exp(-0.5 * ((angles / self.exploration_sigma) ** 2))
        exploration_probs[exploration_probs.isnan()] = 1.  # what to do with 0 velocities?
        if action_probs.size(-1) == 1:  # deterministic policy
            # normalize
            exploration_probs = exploration_probs.scatter(-1, action_probs, 0.)
            exploration_probs /= exploration_probs.sum(-1, keepdim=True)
            # insert no-exploration prob
            exploration_probs *= self.exploration_epsilon
            exploration_probs = exploration_probs.scatter(-1, action_probs, 1 - self.exploration_epsilon)
        else:  # stochastic policy
            exploration_probs /= exploration_probs.sum(-1, keepdim=True)
            exploration_probs = self.exploration_epsilon * exploration_probs + (1 - self.exploration_epsilon) * action_probs
        return exploration_probs

    def _post_update(self, state_dict):
        if self.train:
            self.action_space.update_goal_vector(state_dict["goal"] - state_dict["robot_position"])
            # explore
            self.exploration_probs = self.get_exploration_probs(self.action_index, state_dict["robot_velocity"])
            if "success_ratio" in self.metrics:
                #if self.best_success_ratio is not None and (self.metrics["success_ratio"] > self.best_success_ratio or self.metrics["success_ratio"] >= self.good_success_ratio):
                if self.best_success_ratio != self.metrics["success_ratio"]:
                    self.exploration_epsilon *= self.exploration_decay
                self.best_success_ratio = self.metrics["success_ratio"]
            self.action_index = torch.distributions.Categorical(probs=self.exploration_probs).sample().unsqueeze(-1)
            if self.epoch % self.log_interval == 0:
                self.summary_writer.add_scalar("exploration/epsilon", self.exploration_epsilon, self.epoch)
        elif self.action_index.size(-1) > 1:
            self.action_index = torch.distributions.Categorical(probs=self.exploration_probs).sample().unsqueeze(-1)
        # get actual action vector
        self.action = self.action_space.get_action(self.action_index)


class ContinuesRLContext(RLContext):
    def __init__(self,
                 # exploration_angle_sigma,
                 # exploration_bb_rep_p,
                 # exploration_magnitude_sigma,
                 # exploration_max_goal_p,
                 # exploration_decay,
                 # exploration_duration,
                 **kwargs):
        super().__init__(**kwargs)
        # self.exploration_angle_sigma = exploration_angle_sigma
        # self.exploration_rot_axis = None
        # self.exploration_angle = 0
        # self.exploration_bb_rep_p = exploration_bb_rep_p
        # self.exploration_bb_rep_dims = np.zeros(3)
        # self.exploration_magnitude_sigma = exploration_magnitude_sigma
        # self.exploration_magnitude = 0
        # self.exploration_goal_p = exploration_max_goal_p
        # self.exploration_decay = exploration_decay
        # self.exploration_duration = int(exploration_duration * 1000 / self.interval_duration)
        # self.exploration_index = 0
        # self.exploration_rot_axis_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3))
        #self.uniform_dist = torch.distributions.uniform.Uniform(0, 1)

    @abstractmethod
    def _get_state_dict_(self):
        return

    @abstractmethod
    def _load_impl_(self, state_dict):
        return

    @abstractmethod
    def _update_impl(self, state_dict, reward):
        return

    def _get_state_dict(self):
        return {**self._get_state_dict_(),
                # "exploration_angle_sigma": self.exploration_angle_sigma,
                # "exploration_bb_rep_p": self.exploration_bb_rep_p,
                # "exploration_magnitude_sigma": self.exploration_magnitude_sigma,
                #"exploration_goal_p": self.exploration_goal_p
                }

    def _load_impl(self, state_dict):
        # self.exploration_angle_sigma = state_dict["exploration_angle_sigma"]
        # self.exploration_bb_rep_p = state_dict["exploration_bb_rep_p"]
        # self.exploration_magnitude_sigma = state_dict["exploration_magnitude_sigma"]
        # self.exploration_goal_p = state_dict["exploration_goal_p"]
        self._load_impl_(state_dict)

    def _post_update(self, state_dict):
        nans = torch.isnan(self.action)
        if torch.any(nans):
            # rospy.logwarn(f"NaNs in action tensor. Epoch {self.epoch}")
            self.action = torch.where(nans, 0, self.action)
        # if self.train:
        #    # explore
        #    explore = state_dict["is_timeout"].logical_not()
        #    if self.exploration_index == 0 or self.exploration_rot_axis is None:
        #        self.exploration_rot_axis = self.exploration_rot_axis_dist.sample([self.robot_batch]).to(DEVICE)
        #        self.exploration_angle = torch.deg2rad(torch.distributions.normal.Normal(loc=0, scale=self.exploration_angle_sigma).sample([self.robot_batch])).unsqueeze(-1).to(
        #            DEVICE) if self.exploration_angle_sigma > EPSILON else torch.zeros([self.robot_batch, 1])
        #        self.exploration_magnitude = torch.distributions.normal.Normal(loc=0, scale=self.exploration_magnitude_sigma).sample([self.robot_batch]).unsqueeze(-1).to(
        #            DEVICE) if self.exploration_magnitude_sigma > EPSILON else torch.zeros([self.robot_batch, 1])
        #        # set repulsion vector
        #        self.exploration_bb_rep_dims = torch.empty_like(self.action)
        #        bb_rep = self.uniform_dist.sample(explore.shape).to(DEVICE) < self.exploration_bb_rep_dims
        #        bb_origin = state_dict["workspace_bb_origin"]
        #        bb_end = bb_origin + state_dict["workspace_bb_dims"]
        #        ee_position = state_dict["robot_position"]
        #        mask = ee_position - BB_REPULSION_DISTANCE < bb_origin
        #        self.exploration_bb_rep_dims = torch.where(bb_rep.logical_and(mask), self.max_force, 0)
        #        mask = mask.logical_not().logical_and(ee_position + BB_REPULSION_DISTANCE > bb_end)
        #        self.exploration_bb_rep_dims = torch.where(bb_rep.logical_and(mask), -self.max_force, self.exploration_bb_rep_dims)
        #    self.exploration_index = (self.exploration_index + 1) % self.exploration_duration
        #    # choose action towards goal with some p
        #    move_to_goal = self.uniform_dist.sample(explore.shape).to(DEVICE) < self.exploration_goal_p
        #    self.exploration_goal_p *= self.exploration_decay
        #    goal_direction = state_dict["goal"] - state_dict["robot_position"]
        #    distance_to_goal = torch.linalg.norm(goal_direction)
        #    goal_direction = goal_direction / distance_to_goal * torch.tensor(self.max_force)
        #    action = torch.where(move_to_goal, goal_direction, self.action)
        #    mask = explore.logical_and(move_to_goal.logical_not())
        #    # change dimensions for which we repulse from the bb wall
        #    action = torch.where((self.exploration_bb_rep_dims != 0).logical_and(mask), self.exploration_bb_rep_dims, action)
        #    self.exploration_bb_rep_p *= self.exploration_decay
        #    # rotate the action vector a bit
        #    self.exploration_rot_axis[:, 2] = - (self.exploration_rot_axis[:, 0] * action[:, 0] + self.exploration_rot_axis[:, 1] * action[:, 1]) / action[:, 2]  # make it perpendicular to the action vector
        #    self.exploration_rot_axis /= torch.linalg.norm(self.exploration_rot_axis, -1)
        #    self.exploration_angle_sigma *= self.exploration_decay
        #    cos_angle = torch.cos(self.exploration_angle)
        #    action = torch.where(mask,
        #                         action * cos_angle +  # Rodrigues' rotation formula
        #                         torch.sin(self.exploration_angle) * torch.linalg.cross(action, self.exploration_rot_axis) +
        #                         self.exploration_rot_axis * torch.linalg.vecdot(self.exploration_rot_axis, action).unsqueeze(-1) * (1 - cos_angle),
        #                         action)
        #    self.exploration_rot_axis = torch.where(self.exploration_angle < 0, self.exploration_rot_axis * -1, self.exploration_rot_axis)
        #    # change magnitude
        #    magnitude = torch.linalg.norm(action)
        #    clipped_magnitude = torch.clip(magnitude + self.exploration_magnitude, 0., self.max_force)
        #    self.summary_writer.add_scalar("magnitude", clipped_magnitude.mean(), self.epoch)
        #    action = torch.where(mask, action / magnitude * clipped_magnitude, action)
        #    self.exploration_magnitude_sigma *= self.exploration_decay
        #    if self.epoch % self.log_interval == 0:
        #        self.summary_writer.add_scalar("exploration/angle_sigma", self.exploration_angle_sigma, self.epoch)
        #        self.summary_writer.add_scalar("exploration/magnitude_sigma", self.exploration_magnitude_sigma, self.epoch)
        #        self.summary_writer.add_scalar("exploration/goal_p", self.exploration_goal_p, self.epoch)
