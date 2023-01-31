import os
import re
import math
import torch
import numpy as np
import rospy
from enum import Enum
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, defaultdict

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
    def __init__(self, fmax, interval_duration, dc, mc, motion_penalty, timeout_penalty, max_penalty, dg, rg):
        self.fmax = torch.tensor(float(fmax), device=DEVICE)
        self.dc = torch.tensor(float(dc), device=DEVICE)
        self.mc = torch.tensor(float(mc), device=DEVICE)
        self.motion_penalty = motion_penalty
        self.timeout_penalty = timeout_penalty
        self.max_penalty = torch.tensor(float(max_penalty), device=DEVICE)
        self.dg = torch.tensor(float(dg), device=DEVICE)
        self.rg = torch.tensor(float(rg), device=DEVICE)
        self.interval_duration = float(interval_duration)

    def __call__(self, state_dict, last_state_dict, mask):
        goal = state_dict["goal"]
        if last_state_dict is None:
            out = torch.full([goal.size(0)], torch.nan)
            return out, out, out, out
        robot_position = state_dict["robot_position"]
        # last_robot_position = last_state_dict["robot_position"]
        collision_distances = (state_dict["collision_distances"][:, x:x + 1] for x in range(state_dict["collision_distances"].size(-1)))
        distance_to_goal = torch.linalg.norm(goal - robot_position, dim=-1).unsqueeze(-1)
        # motion_reward = torch.where(mask, (torch.linalg.norm(goal - last_robot_position) - distance_to_goal) / (self.fmax * self.interval_duration), torch.nan)
        collision_penalty = torch.where(mask, 0, torch.nan)
        motion_reward = torch.where(mask, torch.full_like(collision_penalty, self.motion_penalty), torch.nan)
        for collision_distance in collision_distances:
            collision_penalty = torch.where(collision_distance.isnan(), 0, collision_penalty + (self.dc / (collision_distance + EPSILON)) ** self.mc)
        collision_penalty = torch.minimum(collision_penalty, self.max_penalty)
        goal_reward = torch.where(mask, torch.where(distance_to_goal > self.dg, torch.zeros_like(collision_penalty), self.rg), torch.nan)
        goal_reward = torch.where(state_dict["is_timeout"], self.timeout_penalty, goal_reward)
        total_reward = motion_reward + collision_penalty + goal_reward
        return total_reward, motion_reward, collision_penalty, goal_reward


class ActionSpace:
    def __init__(self, grid_order, magnitude_order, max_force):
        assert grid_order > 0
        assert magnitude_order > 0
        angle_step = math.pi / 2 / grid_order
        theta_step_num = 2 * grid_order
        phi_step_num = 4 * grid_order
        self.magnitude_step = max_force / magnitude_order
        self.action_vectors = [torch.zeros([1, 3], device=DEVICE)]
        self.action_vectors_normalized = [None]
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
        for i in range(1, len(self.action_vectors), len(self.goal_vectors)):
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
                 state_augmenter,
                 output_directory,
                 save_rate,
                 interval_duration,
                 log=True,
                 **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.robot_batch = robot_batch
        self.max_force = float(max_force)
        self.output_dir = output_directory
        self.save_rate = save_rate
        self.summary_writer = SummaryWriter(os.path.join(output_directory, "logs"), max_queue=10000, flush_secs=10)
        self.log_interval = 10000
        self.log_dict = {"reward": 0}
        self.episode_accumulators = defaultdict(RLContext.AccumulatorFactory(robot_batch))
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
        self.interval_duration = interval_duration
        self.stop_update = False
        self.state_batch = None
        self.velocity_batch = None
        self.action_batch = None
        self.reward_batch = None
        self.successes = 0
        self.collisions = 0
        self.episodes_passed = 0
        self.timeouts = 0
        self.log = log

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

    def update(self, state_dict):
        goal = state_dict["goal"]
        if self.goal is None:
            self.goal = goal
        # check for finished episodes
        is_new_episode = goal.not_equal(self.goal).any(-1, True)
        is_not_new_episode = is_new_episode.logical_not()
        if is_new_episode.any():
            for key in self.episode_accumulators:
                for i, value in enumerate(self.episode_accumulators[key].get_value(is_new_episode)):
                    self.summary_writer.add_scalar(key, value, self.total_episode_count + i)
                self.episode_accumulators[key].reset(is_new_episode)
            steps_per_episode = torch.masked_select(self.epoch - self.episode_start, is_new_episode).cpu().numpy()
            for value in steps_per_episode:
                self.summary_writer.add_scalar("steps_per_episode", value, self.total_episode_count)
                self.total_episode_count += 1
            self.goal = goal
            self.episode_start = torch.where(is_new_episode, self.epoch, self.episode_start)
        # get reward
        reward, motion_reward, collision_penalty, goal_reward = self.reward_function(state_dict, self.last_state_dict, is_not_new_episode)
        total_reward_mean = torch.masked_select(reward, torch.logical_not(reward.isnan())).mean().cpu()
        if not torch.isnan(total_reward_mean):
            self.log_dict["reward"] += total_reward_mean
        self.summary_writer.add_scalar("reward/per_epoch/total", total_reward_mean, self.epoch)
        self.episode_accumulators["reward/per_episode/total"].update_state(reward, is_not_new_episode)
        self.summary_writer.add_scalar("reward/per_epoch/motion", torch.masked_select(motion_reward, torch.logical_not(motion_reward.isnan())).mean(), self.epoch)
        self.episode_accumulators["reward/per_episode/motion"].update_state(motion_reward, is_not_new_episode)
        self.summary_writer.add_scalar("reward/per_epoch/collision", torch.masked_select(collision_penalty, torch.logical_not(collision_penalty.isnan())).mean(), self.epoch)
        self.episode_accumulators["reward/per_episode/collision"].update_state(collision_penalty, is_not_new_episode)
        self.summary_writer.add_scalar("reward/per_epoch/goal/", torch.masked_select(goal_reward, torch.logical_not(goal_reward.isnan())).mean(), self.epoch)
        self.episode_accumulators["reward/per_episode/goal"].update_state(goal_reward, is_not_new_episode)
        # do the update
        self._update_impl(state_dict, reward)
        self._post_update(state_dict)
        self.last_state_dict = state_dict
        # log
        if self.epoch > 0:
            self.successes += state_dict["reached_goal"].sum().item()
            self.collisions += state_dict["collided"].sum().item()
            self.timeouts = state_dict["is_timeout"].sum().item()
            self.episodes_passed += state_dict["is_terminal"].sum().item()
            if self.epoch % self.log_interval == 0:
                self.summary_writer.add_scalar("metrics/success_ratio", self.successes / (self.episodes_passed + EPSILON), self.epoch)
                self.summary_writer.add_scalar("metrics/collision_ratio", self.collisions / (self.episodes_passed + EPSILON), self.epoch)
                self.summary_writer.add_scalar("metrics/timeout_ratio", self.timeouts / (self.episodes_passed + EPSILON), self.epoch)
                self.successes = self.collisions = self.timeouts = self.episodes_passed = 0
                if self.log:
                    string = f"Epoch {self.epoch} | "
                    for key, value in self.log_dict.items():
                        string += f"{key}: {value / self.log_interval}\t "
                        self.log_dict[key] = 0
                    rospy.loginfo(string)
            if self.epoch % self.save_rate == 0:
                self.save()
        self.epoch += 1
        return self.action

    def warn(self, string):
        if self.log:
            rospy.warn(string)


class DiscreteRLContext(RLContext):
    def __init__(self, grid_order, magnitude_order, exploration_epsilon, exploration_goal_p, exploration_decay, **kwargs):
        super().__init__(**kwargs)
        self.action_space = ActionSpace(grid_order, magnitude_order, self.max_force)
        self.exploration_probs = None
        self.exploration_epsilon = exploration_epsilon
        self.exploration_goal_p = exploration_goal_p
        self.exploration_decay = exploration_decay
        self.action_index = None

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
                "exploration_epsilon": self.exploration_epsilon}

    def _load_impl(self, state_dict):
        self.exploration_epsilon = state_dict["exploration_epsilon"]
        self._load_impl_(state_dict)

    def _post_update(self, state_dict):
        self.action_space.update_goal_vector(state_dict["goal"] - state_dict["robot_position"])
        # explore
        self.exploration_probs = torch.full([self.robot_batch, len(self.action_space)], torch.nan)
        n = torch.zeros_like(self.action_index)
        for i in range(len(self.action_space.goal_vectors)):
            n = torch.where(self.action_space.goal_vectors_index_start + i == self.action_index, n, n + 1)
        for i in range(len(self.action_space.goal_vectors)):
            self.exploration_probs = self.exploration_probs.scatter(-1, self.action_space.goal_vectors_index_start + i, self.exploration_epsilon * self.exploration_goal_p / n)
        self.exploration_probs = self.exploration_probs.scatter(-1, self.action_index, 1 - self.exploration_epsilon)
        mask = self.exploration_probs.isnan()
        n = mask.sum(-1, keepdims=True)
        self.exploration_probs = torch.where(mask, self.exploration_epsilon * (1 - self.exploration_goal_p) / n, self.exploration_probs)
        self.summary_writer.add_scalar("exploration/epsilon", self.exploration_epsilon, self.epoch)
        self.exploration_epsilon *= self.exploration_decay
        self.action_index = torch.distributions.Categorical(self.exploration_probs).sample().unsqueeze(-1)
        # get actual action vector
        self.action = self.action_space.get_action(self.action_index)


class ContinuesRLContext(RLContext):
    def __init__(self,
                 exploration_angle_sigma,
                 exploration_bb_rep_p,
                 exploration_magnitude_sigma,
                 exploration_goal_p,
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
        self.exploration_goal_p = exploration_goal_p
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
        self.summary_writer.add_scalar("exploration/angle_sigma", self.exploration_angle_sigma, self.epoch)
        self.summary_writer.add_scalar("exploration/magnitude_sigma", self.exploration_magnitude_sigma, self.epoch)
        self.summary_writer.add_scalar("exploration/goal_p", self.exploration_goal_p, self.epoch)
