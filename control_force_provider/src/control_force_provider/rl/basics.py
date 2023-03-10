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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-9
BB_REPULSION_DISTANCE = 1e-2

Transition = namedtuple('Transition', ('state', 'velocity', 'action', 'next_state', 'reward'))


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
        if last_state_dict is None:
            out = torch.full([goal.size(0), 1], torch.nan, device=DEVICE)
            return out, out, out, out
        robot_position = state_dict["robot_position"]
        # last_robot_position = last_state_dict["robot_position"]
        collision_distances = (state_dict["collision_distances"][:, x:x + 1] for x in range(state_dict["collision_distances"].size(-1)))
        distance_to_goal = torch.linalg.norm(goal - robot_position, dim=-1).unsqueeze(-1)
        # motion_reward = torch.where(mask, (torch.linalg.norm(goal - last_robot_position) - distance_to_goal) / (self.fmax * self.interval_duration), torch.nan)
        collision_penalty = torch.where(mask, 0, torch.nan)
        motion_reward = torch.where(mask, torch.full_like(collision_penalty, self.motion_penalty), torch.nan)
        for collision_distance in collision_distances:
            collision_penalty = -torch.where(collision_distance.isnan(), 0, collision_penalty + (self.dc / (collision_distance + EPSILON)) ** self.mc)
        collision_penalty = torch.minimum(collision_penalty, self.max_penalty)
        goal_reward = torch.where(mask, torch.where(state_dict["reached_goal"], self.rg, torch.zeros_like(collision_penalty)), torch.nan)
        if self.max_distance is None:
            self.max_distance = torch.linalg.norm(state_dict["workspace_bb_dims"])
        timeout_penalty = self.min_timeout_penalty + ((distance_to_goal / self.max_distance) * self.timeout_penalty_range)
        goal_reward = torch.where(state_dict["is_timeout"], timeout_penalty, goal_reward)
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
        index = index.expand([-1, 3])
        actions = torch.gather(self.action_space_tensor, 0, index)
        for i in range(len(self.goal_vectors)):
            actions = torch.where(index == self.goal_vectors_index_start + i, self.goal_vectors[i], actions)
        return actions

    def __len__(self):
        return len(self.action_vectors)


class RLContext(ABC):
    class Accumulator:
        def __init__(self, batch_size=1):
            self.state = torch.zeros(batch_size, device=DEVICE)
            self.count = torch.zeros(batch_size, device=DEVICE)
            self.default_mask = torch.ones_like(self.state, device=DEVICE, dtype=torch.bool)

        def update_state(self, value, mask=None):
            if mask is None:
                mask = self.default_mask
            mask = mask.to(DEVICE)
            value = value.to(DEVICE)
            self.state = torch.where(mask, self.state + value, self.state)
            self.count = torch.where(mask, self.count + 1, self.count)

        def get_value(self, mask=None):
            if mask is None:
                mask = self.default_mask
            value = torch.where(torch.logical_and(mask, self.count != 0), self.state / self.count, 0)
            return torch.masked_select(value, mask).cpu().numpy()

        def reset(self, mask=None):
            if mask is None:
                mask = self.default_mask
            self.state = torch.where(mask, 0, self.state)
            self.count = torch.where(mask, 0, self.count)

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
                 state_augmenter,
                 output_directory,
                 save_rate,
                 interval_duration,
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
        self.log_interval = 10000
        self.log_dict = {}
        self.episode_accumulators = defaultdict(RLContext.AccumulatorFactory(robot_batch))
        self.episode_accumulators_mean = {}
        self.reward_function = reward_function
        self.state_augmenter = state_augmenter
        self.last_state_dict = None
        self.action = None
        self.epoch = 0
        self.episode = torch.zeros((robot_batch, 1), device=DEVICE)
        self.episode_start = torch.zeros((robot_batch, 1), device=DEVICE)
        self.total_episode_count = 0
        self.last_episode_count = 0
        self.goal = None
        self.goal_reached_threshold_distance = goal_reached_threshold_distance
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
        self.acc_reward = torch.zeros([robot_batch, 1], device=DEVICE)
        self.acc_lengths = torch.zeros([robot_batch, 1], device=DEVICE)
        self.log = log
        self.train = train
        self.evaluating = False
        self.evaluation_duration = evaluation_duration
        self.evaluation_epoch = 0
        self.metrics = {}
        self.her_reward = self.reward_function.rg + self.reward_function.motion_penalty
        self.her_noise_dist = self.noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * 1e-7)
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
            self.episode_start = torch.full_like(self.episode_start, self.epoch)
            self.total_episode_count = self.last_episode_count = state_dict["episode"]
            self._load_impl(state_dict)

    @staticmethod
    def _copy_state_dict(state_dict):
        state_dict_copy = {}
        for key, value in state_dict.items():
            state_dict_copy[key] = value.clone()
        return state_dict_copy

    def update(self, state_dict):
        reward = None
        if self.train:
            if self.max_distance is None:
                self.max_distance = torch.linalg.norm(state_dict["workspace_bb_dims"])
            goal = state_dict["goal"]
            if self.goal is None:
                self.goal = goal
            # check for finished episodes
            is_new_episode = goal.not_equal(self.goal).any(-1, True)
            is_not_new_episode = is_new_episode.logical_not()
            if is_new_episode.any():
                for key in self.episode_accumulators:
                    value = self.episode_accumulators[key].get_value(is_new_episode)
                    if key not in self.episode_accumulators_mean:
                        self.episode_accumulators_mean[key] = value if key not in self.episode_accumulators_mean else self.episode_accumulators_mean[key] + value
                    self.episode_accumulators[key].reset(is_new_episode)
                steps_per_episode = torch.masked_select(self.epoch - self.episode_start, is_new_episode).cpu().numpy()
                self.total_episode_count += is_new_episode.sum().item()
                self.goal = goal
                self.episode_start = torch.where(is_new_episode, self.epoch, self.episode_start)
            # get reward
            reward, motion_reward, collision_penalty, goal_reward = self.reward_function(state_dict, self.last_state_dict, is_not_new_episode)
            self.acc_reward = torch.where(reward.isnan(), self.acc_reward, self.acc_reward + reward)
            mean_return = torch.masked_select(self.acc_reward, state_dict["is_terminal"]).mean()
            if not mean_return.isnan():
                self.returns.append(mean_return.item())
            self.acc_reward = torch.where(state_dict["is_terminal"], 0, self.acc_reward)
        # do the update
        self._update_impl(state_dict, reward)
        self._post_update(state_dict)
        self.last_state_dict = self._copy_state_dict(state_dict)
        if self.evaluating:
            self.acc_lengths += 1
            self.evaluation_epoch += 1
            self.successes.append(state_dict["reached_goal"].sum().item())
            self.collisions.append(state_dict["collided"].sum().item())
            self.timeouts.append(state_dict["is_timeout"].sum().item())
            self.episodes_passed.append(state_dict["is_terminal"].sum().item())
            self.episodes_lengths.append(self.acc_lengths[state_dict["is_terminal"]].sum().item())
            self.acc_lengths = torch.where(state_dict["is_terminal"], 0, self.acc_lengths)
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
                self.metrics["max_episode_length"] = max(episodes_lengths)
                self.metrics["mean_episode_length"] = sum(episodes_lengths) / (episodes_passed + EPSILON)
                for key in self.metrics:
                    self.summary_writer.add_scalar(f"metrics/{key}", self.metrics[key], self.epoch)
                for key in self.episode_accumulators_mean:
                    self.summary_writer.add_scalar(key, sum(self.episode_accumulators_mean[key] / len(self.episode_accumulators_mean), self.epoch))
                if self.log:
                    string = f"Epoch {self.epoch} | "
                    for key, value in self.log_dict.items():
                        mean = value / self.log_interval
                        self.summary_writer.add_scalar(key, mean, self.epoch)
                        string += f"{key}: {mean}\t "
                        self.log_dict[key] = 0
                    string += f"return: {mean_return}"
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
                self.acc_reward = torch.zeros([self.robot_batch, 1], device=DEVICE)
                self.acc_lengths = torch.zeros([self.robot_batch, 1], device=DEVICE)
        else:
            self.epoch += 1
            if self.epoch > 0:
                if self.epoch % self.log_interval == 0:
                    self.evaluating = True
                    self.train = False
                if self.epoch % self.save_rate == 0:
                    self.save()
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

    def _post_update(self, state_dict):
        if self.train:
            self.action_space.update_goal_vector(state_dict["goal"] - state_dict["robot_position"])
            # explore
            # exploration probs depend on the angle to the current velocity (normal distribution)
            velocities_normalized = state_dict["robot_velocity"] / (torch.linalg.vector_norm(state_dict["robot_velocity"], dim=-1, keepdim=True) + EPSILON)
            angles = torch.matmul(velocities_normalized.view([self.robot_batch, 1, 3]), self.action_vectors)
            angles = torch.acos(torch.clamp(angles, -1, 1)).squeeze()
            self.exploration_probs = self.exploration_gauss_factor * torch.exp(-0.5 * ((angles / self.exploration_sigma) ** 2))
            self.exploration_probs[self.exploration_probs.isnan()] = 0.01  # what to do with 0 velocities?
            # normalize
            self.exploration_probs = self.exploration_probs.scatter(-1, self.action_index, 0)
            self.exploration_probs /= (self.exploration_probs.sum(-1, keepdim=True))
            # insert no-exploration prob
            self.exploration_probs *= self.exploration_epsilon
            self.exploration_probs = self.exploration_probs.scatter(-1, self.action_index, 1 - self.exploration_epsilon)
            if "success_ratio" in self.metrics:
                if self.best_success_ratio is not None and (self.metrics["success_ratio"] > self.best_success_ratio or self.metrics["success_ratio"] >= self.good_success_ratio):
                    self.exploration_epsilon *= self.exploration_decay
                self.best_success_ratio = max(self.metrics["success_ratio"], self.best_success_ratio)
            self.action_index = torch.distributions.Categorical(self.exploration_probs).sample().unsqueeze(-1)
            if self.epoch % self.log_interval == 0:
                self.summary_writer.add_scalar("exploration/epsilon", self.exploration_epsilon, self.epoch)
        # get actual action vector
        self.action = self.action_space.get_action(self.action_index)


class ContinuesRLContext(RLContext):
    def __init__(self,
                 exploration_angle_sigma,
                 exploration_bb_rep_p,
                 exploration_magnitude_sigma,
                 exploration_max_goal_p,
                 exploration_decay,
                 exploration_duration,
                 **kwargs):
        super().__init__(**kwargs)
        self.exploration_angle_sigma = exploration_angle_sigma
        self.exploration_rot_axis = None
        self.exploration_angle = 0
        self.exploration_bb_rep_p = exploration_bb_rep_p
        self.exploration_bb_rep_dims = np.zeros(3)
        self.exploration_magnitude_sigma = exploration_magnitude_sigma
        self.exploration_magnitude = 0
        self.exploration_goal_p = exploration_max_goal_p
        self.exploration_decay = exploration_decay
        self.exploration_duration = int(exploration_duration * 1000 / self.interval_duration)
        self.exploration_index = 0
        self.exploration_rot_axis_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3))
        self.uniform_dist = torch.distributions.uniform.Uniform(0, 1)

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
                "exploration_angle_sigma": self.exploration_angle_sigma,
                "exploration_bb_rep_p": self.exploration_bb_rep_p,
                "exploration_magnitude_sigma": self.exploration_magnitude_sigma,
                "exploration_goal_p": self.exploration_goal_p}

    def _load_impl(self, state_dict):
        self.exploration_angle_sigma = state_dict["exploration_angle_sigma"]
        self.exploration_bb_rep_p = state_dict["exploration_bb_rep_p"]
        self.exploration_magnitude_sigma = state_dict["exploration_magnitude_sigma"]
        self.exploration_goal_p = state_dict["exploration_goal_p"]
        self._load_impl_(state_dict)

    def _post_update(self, state_dict):
        nans = torch.isnan(self.action)
        if torch.any(nans):
            # rospy.logwarn(f"NaNs in action tensor. Epoch {self.epoch}")
            self.action = torch.where(nans, 0, self.action)
        if self.train:
            # explore
            explore = state_dict["is_timeout"].logical_not()
            if self.exploration_index == 0 or self.exploration_rot_axis is None:
                self.exploration_rot_axis = self.exploration_rot_axis_dist.sample([self.robot_batch]).to(DEVICE)
                self.exploration_angle = torch.deg2rad(torch.distributions.normal.Normal(loc=0, scale=self.exploration_angle_sigma).sample([self.robot_batch])).unsqueeze(-1).to(
                    DEVICE) if self.exploration_angle_sigma > EPSILON else torch.zeros([self.robot_batch, 1])
                self.exploration_magnitude = torch.distributions.normal.Normal(loc=0, scale=self.exploration_magnitude_sigma).sample([self.robot_batch]).unsqueeze(-1).to(
                    DEVICE) if self.exploration_magnitude_sigma > EPSILON else torch.zeros([self.robot_batch, 1])
                # set repulsion vector
                self.exploration_bb_rep_dims = torch.empty_like(self.action)
                bb_rep = self.uniform_dist.sample(explore.shape).to(DEVICE) < self.exploration_bb_rep_dims
                bb_origin = state_dict["workspace_bb_origin"]
                bb_end = bb_origin + state_dict["workspace_bb_dims"]
                ee_position = state_dict["robot_position"]
                mask = ee_position - BB_REPULSION_DISTANCE < bb_origin
                self.exploration_bb_rep_dims = torch.where(bb_rep.logical_and(mask), self.max_force, 0)
                mask = mask.logical_not().logical_and(ee_position + BB_REPULSION_DISTANCE > bb_end)
                self.exploration_bb_rep_dims = torch.where(bb_rep.logical_and(mask), -self.max_force, self.exploration_bb_rep_dims)
            self.exploration_index = (self.exploration_index + 1) % self.exploration_duration
            # choose action towards goal with some p
            move_to_goal = self.uniform_dist.sample(explore.shape).to(DEVICE) < self.exploration_goal_p
            self.exploration_goal_p *= self.exploration_decay
            goal_direction = state_dict["goal"] - state_dict["robot_position"]
            distance_to_goal = torch.linalg.norm(goal_direction)
            goal_direction = goal_direction / distance_to_goal * torch.tensor(self.max_force)
            action = torch.where(move_to_goal, goal_direction, self.action)
            mask = explore.logical_and(move_to_goal.logical_not())
            # change dimensions for which we repulse from the bb wall
            action = torch.where((self.exploration_bb_rep_dims != 0).logical_and(mask), self.exploration_bb_rep_dims, action)
            self.exploration_bb_rep_p *= self.exploration_decay
            # rotate the action vector a bit
            self.exploration_rot_axis[:, 2] = - (self.exploration_rot_axis[:, 0] * action[:, 0] + self.exploration_rot_axis[:, 1] * action[:, 1]) / action[:, 2]  # make it perpendicular to the action vector
            self.exploration_rot_axis /= torch.linalg.norm(self.exploration_rot_axis, -1)
            self.exploration_angle_sigma *= self.exploration_decay
            cos_angle = torch.cos(self.exploration_angle)
            action = torch.where(mask,
                                 action * cos_angle +  # Rodrigues' rotation formula
                                 torch.sin(self.exploration_angle) * torch.linalg.cross(action, self.exploration_rot_axis) +
                                 self.exploration_rot_axis * torch.linalg.vecdot(self.exploration_rot_axis, action).unsqueeze(-1) * (1 - cos_angle),
                                 action)
            self.exploration_rot_axis = torch.where(self.exploration_angle < 0, self.exploration_rot_axis * -1, self.exploration_rot_axis)
            # change magnitude
            magnitude = torch.linalg.norm(action)
            clipped_magnitude = torch.clip(magnitude + self.exploration_magnitude, 0., self.max_force)
            self.summary_writer.add_scalar("magnitude", clipped_magnitude.mean(), self.epoch)
            action = torch.where(mask, action / magnitude * clipped_magnitude, action)
            self.exploration_magnitude_sigma *= self.exploration_decay
            if self.epoch % self.log_interval == 0:
                self.summary_writer.add_scalar("exploration/angle_sigma", self.exploration_angle_sigma, self.epoch)
                self.summary_writer.add_scalar("exploration/magnitude_sigma", self.exploration_magnitude_sigma, self.epoch)
                self.summary_writer.add_scalar("exploration/goal_p", self.exploration_goal_p, self.epoch)
