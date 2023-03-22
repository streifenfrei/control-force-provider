import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from threading import Lock
import time
from .basics import *


class ReplayBuffer:
    def __init__(self, capacity, transition=Transition):
        self.buffer = [None for _ in transition._fields]
        self.transition = transition
        self.capacity = capacity
        self.lock = Lock()

    def push(self, *args):
        args = list(arg.to(DEVICE) for arg in args)
        t = self.transition(*args)
        mask = torch.zeros_like(t.reward)
        for field in t._fields:
            if field not in ["next_state"]:
                mask = getattr(t, field).isnan().any(-1, keepdims=True).logical_or(mask)
        mask = mask.logical_not()
        args = []
        for v in t:
            args.append(v[mask.expand([-1, v.size(-1)])].view([-1, v.size(-1)]).cpu())
        with self.lock:
            for i, arg in enumerate(args):
                self.buffer[i] = arg if self.buffer[i] is None else torch.cat([arg, self.buffer[i]])
                if self.buffer[i].size(0) > self.capacity:
                    self.buffer[i] = self.buffer[i][:self.capacity, :]

    def sample(self, batch_size):
        def generator():
            start_index = 0
            end_index = batch_size
            while start_index < len(self):
                args = []
                with self.lock:
                    for buffer in self.buffer:
                        args.append(buffer[start_index:end_index, :].to(DEVICE))
                yield self.transition(*args)
                start_index += batch_size
                end_index = min(len(self) - 1, end_index + batch_size)
        return generator

    def __len__(self):
        return 0 if self.buffer[0] is None else len(self.buffer[0])


class DQN(nn.Module):
    def __init__(self, state_dim, out_dim, layer_size):
        super(DQN, self).__init__()
        self.dense1 = nn.Linear(state_dim, layer_size)
        self.bnorm1 = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.bnorm2 = nn.BatchNorm1d(layer_size)
        self.dense3 = nn.Linear(layer_size, layer_size)
        self.bnorm3 = nn.BatchNorm1d(layer_size)
        self.out = nn.Linear(layer_size, out_dim)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = self.bnorm1(x)
        x = torch.relu(self.dense2(x))
        x = self.bnorm2(x)
        x = torch.relu(self.dense3(x))
        x = self.bnorm3(x)
        return self.out(x)


class DQNContext(DiscreteRLContext):
    def __init__(self,
                 layer_size,
                 replay_buffer_size,
                 target_network_update_rate,
                 **kwargs):
        super().__init__(**kwargs)
        self.dqn_policy = DQN(self.state_dim, len(self.action_space), layer_size).to(DEVICE)
        self.dqn_target = DQN(self.state_dim, len(self.action_space), layer_size).to(DEVICE)
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_target.eval()
        self.optimizer = torch.optim.Adam(self.dqn_policy.parameters())
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.target_network_update_rate = target_network_update_rate
        self.ts_model = os.path.join(self.output_dir, "dqn_ts.pt")
        self.total_loss_accumulator = RLContext.Accumulator()
        self.rl_loss_accumulator = RLContext.Accumulator()
        self.batch_load_time_accumulator = RLContext.Accumulator()
        self.update_future = None
        self.log_dict["loss"] = 0

    def _get_state_dict_(self):
        return {"model_state_dict": self.dqn_policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()}
        # torch.jit.script(self.dqn_policy).save(self.ts_model)

    def _load_impl_(self, state_dict):
        self.dqn_policy.load_state_dict(state_dict["model_state_dict"])
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn_policy.parameters())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def _update_impl(self, state_dict, reward):
        if self.train:
            if self.last_state_dict is not None:
                next_state = torch.where(state_dict["is_terminal"].expand(-1, state_dict["state"].size(-1)), torch.nan, state_dict["state"])
                self.replay_buffer.push(self.last_state_dict["state"], self.last_state_dict["robot_velocity"], self.action_index, next_state, reward)
                # hindsight experience replay
                her_transition = self.create_her_transitions(self.last_state_dict, self.action_index, state_dict, reward)
                if her_transition is not None:
                    self.replay_buffer.push(*her_transition)

            if len(self.replay_buffer) >= self.batch_size:
                data_load_start = time.time()
                batch = self.replay_buffer.sample(self.batch_size)
                state_batch = batch.state
                action_batch = batch.action
                reward_batch = batch.reward
                next_state_batch = batch.next_state
                is_terminal = next_state_batch.isnan().any(-1, keepdims=True)
                next_state_batch = torch.where(next_state_batch.isnan(), 0, next_state_batch)
                self.optimizer.zero_grad()
                q = self.dqn_policy(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    q_target = self.dqn_target(next_state_batch).max(1, keepdims=True)[0]
                q_target_be = self.dqn_policy(next_state_batch).max(1, keepdims=True)[0]
                dqn_target = torch.where(is_terminal, reward_batch, reward_batch + self.discount_factor * q_target)
                be_target = torch.where(is_terminal, reward_batch, reward_batch + self.discount_factor * q_target_be)
                dqn_loss = nn.MSELoss(reduction="none")(q, dqn_target)
                bellman_error = nn.MSELoss(reduction="none")(q, be_target)
                loss = torch.maximum(dqn_loss, bellman_error).mean()
                if not loss.isnan().any():
                    self.total_loss_accumulator.update_state(loss.detach().mean().cpu())
                    loss.backward()
                    clip_grad_norm_(self.dqn_policy.parameters(), 1)
                    self.optimizer.step()
                    self.total_loss_accumulator.update_state(loss.detach().mean().cpu())
                    self.log_dict["loss"] += self.total_loss_accumulator.get_value().item()
                    self.total_loss_accumulator.reset()
                else:
                    rospy.logwarn(f"NaNs in DQN loss. Epoch {self.epoch}")

            if self.epoch % self.target_network_update_rate == 0:
                self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_policy.eval()
        with torch.no_grad():
            self.action_index = self.dqn_policy(state_dict["state"]).max(-1)[1].unsqueeze(-1)
        self.dqn_policy.train()


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


class DQNNAFContext(ContinuesRLContext):
    def __init__(self,
                 layer_size,
                 replay_buffer_size,
                 target_network_update_rate,
                 dot_loss_factor,
                 dot_loss_decay,
                 **kwargs):
        super().__init__(**kwargs)
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

    def _get_state_dict_(self):
        return {"model_state_dict": self.dqn_policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "dot_loss_factor": self.dot_loss_factor}
        # torch.jit.script(self.dqn_policy).save(self.ts_model)

    def _load_impl_(self, state_dict):
        self.dqn_policy.load_state_dict(state_dict["model_state_dict"])
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn_policy.parameters())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.dot_loss_factor = state_dict["dot_loss_factor"]

    def _update_impl(self, state_dict, reward):
        if self.train:
            if self.last_state_dict is not None:
                next_state = torch.where(state_dict["is_terminal"].expand(-1, state_dict["state"].size(-1)), torch.nan, state_dict["state"])
                self.replay_buffer.push(self.last_state_dict["state"], self.last_state_dict["robot_velocity"], self.action, next_state, reward)

            if len(self.replay_buffer) >= self.batch_size:
                data_load_start = time.time()
                batch = self.replay_buffer.sample(self.batch_size)
                state_batch = batch.state
                velocity_batch = batch.velocity
                action_batch = batch.action
                reward_batch = batch.reward
                next_state_batch = batch.next_state
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
                    clip_grad_norm_(self.dqn_policy.parameters(), 100)
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
                    self.warn(f"NaNs in DQN loss. Epoch {self.epoch}")

            if self.epoch % self.target_network_update_rate == 0:
                self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_policy.eval()
        with torch.no_grad():
            self.action = self.dqn_policy(state_dict["state"])[0].squeeze(0)
        self.dqn_policy.train()
