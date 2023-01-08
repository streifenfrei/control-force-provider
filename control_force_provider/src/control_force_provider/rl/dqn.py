import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import random
import time
from collections import deque
from .basics import *


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        chunks = []
        for arg in args:
            chunks.append(arg.cpu().split(1))
        for chunk in zip(*chunks):
            transition = Transition(*chunk)
            if not any(getattr(transition, field).isnan().any() for field in transition._fields if field not in ["next_state"]):
                self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNNAF(nn.Module):
    def __init__(self, state_dim, action_dim, layer_size, max_force):
        super(DQNNAF, self).__init__()
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
        l = torch.zeros((state.shape[0], self.action_dim, self.action_dim)).to(DEVICE)
        indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        l[:, indices[0], indices[1]] = l_entries
        l.diagonal(dim1=1, dim2=2).exp_()
        p = l * l.transpose(2, 1)
        q = None
        if action is not None:
            action_diff = (action - mu).unsqueeze(-1)
            a = (-0.5 * torch.matmul(torch.matmul(action_diff.transpose(2, 1), p), action_diff)).squeeze(-1)
            q = a + v

        return mu, q, v


class DQNNAFContext(RLContext):
    def __init__(self,
                 layer_size,
                 replay_buffer_size,
                 target_network_update_rate,
                 dot_loss_factor,
                 dot_loss_decay,
                 exploration_angle_sigma,
                 exploration_bb_rep_p,
                 exploration_magnitude_sigma,
                 exploration_goal_p,
                 exploration_decay,
                 exploration_duration,
                 **kwargs):
        super().__init__(
            exploration_scheme=ContinuesExploration(kwargs["interval_duration"], kwargs["max_force"], exploration_angle_sigma, exploration_bb_rep_p, exploration_magnitude_sigma, exploration_goal_p, exploration_decay,
                                                    exploration_duration), **kwargs)
        self.dqn_policy = DQNNAF(self.state_dim, self.action_dim, layer_size, self.max_force).to(DEVICE)
        self.dqn_target = DQNNAF(self.state_dim, self.action_dim, layer_size, self.max_force).to(DEVICE)
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_target.eval()
        self.optimizer = torch.optim.Adam(self.dqn_policy.parameters())
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.target_network_update_rate = target_network_update_rate
        self.ts_model = os.path.join(self.output_dir, "dqn_ts.pt")
        self.total_loss_accumulator = RLContext.Accumulator()
        self.rl_loss_accumulator = RLContext.Accumulator()
        self.dot_loss_accumulator = RLContext.Accumulator()
        self.batch_load_time_accumulator = RLContext.Accumulator()
        self.update_future = None
        self.dot_loss_factor = dot_loss_factor
        self.dot_loss_decay = dot_loss_decay
        self.log_dict["loss"] = 0

    def _get_state_dict(self):
        return {"model_state_dict": self.dqn_policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "dot_loss_factor": self.dot_loss_factor}
        # torch.jit.script(self.dqn_policy).save(self.ts_model)

    def _load_impl(self, state_dict):
        self.dqn_policy.load_state_dict(state_dict["model_state_dict"])
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn_policy.parameters())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.dot_loss_factor = state_dict["dot_loss_factor"]

    def _update_impl(self, state_dict, reward, is_terminal):
        if self.last_state_dict is not None:
            next_state = torch.where(is_terminal.expand(-1, state_dict["state"].size(-1)), torch.nan, state_dict["state"])
            self.replay_buffer.push(self.last_state_dict["state"], state_dict["robot_velocity"], self.action, next_state, reward)

        if len(self.replay_buffer) >= self.batch_size:
            data_load_start = time.time()
            batch = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))
            state_batch = torch.cat(batch.state).to(DEVICE)
            velocity_batch = torch.cat(batch.velocity).to(DEVICE)
            action_batch = torch.cat(batch.action).to(DEVICE)
            reward_batch = torch.cat(batch.reward).to(DEVICE)
            next_state_batch = torch.cat(batch.next_state).to(DEVICE)
            is_terminal = next_state_batch.isnan().any(-1, keepdims=True)
            next_state_batch = torch.where(next_state_batch.isnan(), state_batch, next_state_batch)
            self.batch_load_time_accumulator.update_state(torch.tensor(time.time() - data_load_start, device=DEVICE))
            self.optimizer.zero_grad()
            mu, q, _ = self.dqn_policy(state_batch, action_batch)
            with torch.no_grad():
                v_target = self.dqn_target(next_state_batch)[2]
            v_target_be = self.dqn_policy(next_state_batch)[2]
            dqn_target = torch.where(is_terminal, reward_batch, reward_batch + self.discount_factor * v_target)
            be_target = torch.where(is_terminal, reward_batch, reward_batch + self.discount_factor * v_target_be)
            dqn_loss = nn.MSELoss(reduction="none")(q, dqn_target)
            bellman_error = nn.MSELoss(reduction="none")(q, be_target)
            rl_loss = torch.maximum(dqn_loss, bellman_error).mean()
            if not rl_loss.isnan().any():
                self.rl_loss_accumulator.update_state(rl_loss.detach().mean().cpu())
                rl_loss *= (1 - self.dot_loss_factor)
                # TODO fix dot loss
                dot_loss = - torch.mean(
                    torch.bmm((mu / (torch.norm(mu, dim=1).unsqueeze(1) + EPSILON)).unsqueeze(1), (velocity_batch / (torch.norm(velocity_batch, dim=1).unsqueeze(1) + EPSILON)).unsqueeze(2)).squeeze(2))
                self.dot_loss_accumulator.update_state(dot_loss.detach().mean().cpu())
                dot_loss *= self.dot_loss_factor
                loss = rl_loss + dot_loss
                self.total_loss_accumulator.update_state(loss.detach().mean().cpu())
                loss.backward()
                clip_grad_norm_(self.dqn_policy.parameters(), 1)
                self.optimizer.step()
                self.log_dict["loss"] += self.total_loss_accumulator.get_value().item()
                self.summary_writer.add_scalar("loss/rl", self.rl_loss_accumulator.get_value(), self.epoch)
                self.summary_writer.add_scalar("loss/dot", self.dot_loss_accumulator.get_value(), self.epoch)
                self.summary_writer.add_scalar("loss/total", self.total_loss_accumulator.get_value(), self.epoch)
                self.summary_writer.add_scalar("profiling/batch_load_time", self.batch_load_time_accumulator.get_value(), self.epoch)
                self.rl_loss_accumulator.reset()
                self.dot_loss_accumulator.reset()
                self.total_loss_accumulator.reset()
                self.batch_load_time_accumulator.reset()
                self.summary_writer.add_scalar("dot_loss_factor", self.dot_loss_factor, self.epoch)
                self.dot_loss_factor *= self.dot_loss_decay
            else:
                rospy.logwarn(f"NaNs in DQN loss. Epoch {self.epoch}")

        if self.epoch % self.target_network_update_rate == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_policy.eval()
        with torch.no_grad():
            self.action = self.dqn_policy(state_dict["state"])[0].squeeze(0)
        self.dqn_policy.train()
