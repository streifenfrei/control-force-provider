import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import random
import time
import itertools
from collections import deque
from .basics import *


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
        l = torch.zeros((state.shape[0], self.action_dim, self.action_dim)).to(DEVICE)
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


class DQNContext(RLContext):
    def __init__(self, layer_size, replay_buffer_size, target_network_update_rate, dot_loss_factor, dot_loss_decay, **kwargs):
        super().__init__(**kwargs)
        self.dqn_policy = DQN(self.state_dim, self.action_dim, layer_size, self.max_force).to(DEVICE)
        self.dqn_target = DQN(self.state_dim, self.action_dim, layer_size, self.max_force).to(DEVICE)
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

    def _get_state_dict(self):
        return {"model_state_dict": self.dqn_policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "dot_loss_factor": self.dot_loss_factor,
                "replay_buffer": list(self.replay_buffer.buffer)}
        # torch.jit.script(self.dqn_policy).save(self.ts_model)

    def _load_impl(self, state_dict):
        self.dqn_policy.load_state_dict(state_dict["model_state_dict"])
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.dot_loss_factor = state_dict["dot_loss_factor"]
        self.replay_buffer.buffer = deque(state_dict["replay_buffer"], maxlen=self.replay_buffer.buffer.maxlen)

    def _update_thread(self):
        batch_size = min(len(self.replay_buffer), self.batch_size)
        if batch_size >= 2:
            while not self.stop_update:
                data_load_start = time.time()
                if self.state_batch is None:
                    batch = Transition(*zip(*list(self.replay_buffer.buffer)))
                    self.state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32).to(DEVICE)
                    self.velocity_batch = torch.tensor(np.stack(batch.velocity), dtype=torch.float32).to(DEVICE)
                    self.action_batch = torch.tensor(np.stack(batch.action), dtype=torch.float32).to(DEVICE)
                    self.reward_batch = torch.tensor(np.stack(batch.reward), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
                else:
                    if batch_size <= self.batch_size:
                        missing = self.state_batch.shape[0] - batch_size
                        if missing < 0:
                            batch = Transition(*zip(*itertools.islice(self.replay_buffer.buffer, len(self.replay_buffer) + missing, len(self.replay_buffer))))
                            self.state_batch = torch.cat([self.state_batch, torch.tensor(np.stack(batch.state), dtype=torch.float32, device=DEVICE)], 0)
                            self.velocity_batch = torch.cat([self.velocity_batch, torch.tensor(np.stack(batch.velocity), dtype=torch.float32, device=DEVICE)], 0)
                            self.action_batch = torch.cat([self.action_batch, torch.tensor(np.stack(batch.action), dtype=torch.float32, device=DEVICE)], 0)
                            self.reward_batch = torch.cat([self.reward_batch, torch.tensor(np.stack(batch.reward), dtype=torch.float32, device=DEVICE).unsqueeze(-1)], 0)
                self.batch_load_time_accumulator.update_state(time.time() - data_load_start)
                mu, q, _ = self.dqn_policy(self.state_batch, self.action_batch)
                with torch.no_grad():
                    v_target = self.dqn_target(self.state_batch)[2]
                target = self.reward_batch + self.discount_factor * v_target
                rl_loss = nn.MSELoss()(q, target)
                self.rl_loss_accumulator.update_state(rl_loss.detach().cpu().numpy())
                rl_loss *= (1 - self.dot_loss_factor)
                dot_loss = - torch.mean(
                    torch.bmm((mu / (torch.norm(mu, dim=1).unsqueeze(1) + EPSILON)).unsqueeze(1), (self.velocity_batch / (torch.norm(self.velocity_batch, dim=1).unsqueeze(1) + EPSILON)).unsqueeze(2)).squeeze(2))
                self.dot_loss_accumulator.update_state(dot_loss.detach().cpu().numpy())
                dot_loss *= self.dot_loss_factor
                loss = rl_loss + dot_loss
                self.total_loss_accumulator.update_state(loss.detach().cpu().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.dqn_policy.parameters(), 1)
                self.optimizer.step()
            self.summary_writer.add_scalar("loss/rl", self.rl_loss_accumulator.get_value(), self.epoch)
            self.summary_writer.add_scalar("loss/dot", self.dot_loss_accumulator.get_value(), self.epoch)
            self.summary_writer.add_scalar("loss/total", self.total_loss_accumulator.get_value(), self.epoch)
            self.summary_writer.add_scalar("profiling/batch_load_time", self.batch_load_time_accumulator.get_value(), self.epoch)
            self.summary_writer.add_scalar("profiling/batch_size", batch_size, self.epoch)
            self.rl_loss_accumulator.reset()
            self.dot_loss_accumulator.reset()
            self.total_loss_accumulator.reset()
            self.batch_load_time_accumulator.reset()
            self.summary_writer.add_scalar("dot_loss_factor", self.dot_loss_factor, self.epoch)
            self.dot_loss_factor *= self.dot_loss_decay

    def _update_impl(self, state_dict, reward):
        if reward is not None:
            self.replay_buffer.push(self.last_state_dict["state"], state_dict["robot_velocity"][:3], self.action, state_dict["state"], reward)
        if self.update_future is not None:
            self.stop_update = True
            self.update_future.result()
        if self.epoch % self.target_network_update_rate == 0:
            self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_policy.eval()
        with torch.no_grad():
            self.action = self.dqn_policy(torch.tensor(state_dict["state"], dtype=torch.float32).unsqueeze(0).to(DEVICE))[0].squeeze(0).cpu().numpy()
        self.dqn_policy.train()
        self.stop_update = False
        self.update_future = self.thread_executor.submit(self._update_thread)
