import torch.nn as nn
import time
from threading import Thread
from torch.nn.utils import clip_grad_norm_
from .basics import *
from .dqn import ReplayBuffer

ACTransition = namedtuple('Transition', ('state', 'velocity', 'action', 'next_state', 'reward', 'is_terminal', 'exploration_prob'))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, out_dim, layer_size):
        super(ActorCritic, self).__init__()
        self.dense1 = nn.Linear(state_dim, layer_size)
        self.bnorm1 = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.bnorm2 = nn.BatchNorm1d(layer_size)
        self.actor = nn.Linear(layer_size, out_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.critic = nn.Linear(layer_size, 1)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = self.bnorm1(x)
        x = torch.relu(self.dense2(x))
        x = self.bnorm2(x)
        return self.softmax(10 * torch.tanh(self.actor(x))), self.critic(x)


class A2CContext(DiscreteRLContext):
    def __init__(self,
                 layer_size,
                 replay_buffer_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor_critic = ActorCritic(self.state_dim, len(self.action_space), layer_size).to(DEVICE)
        self.kld = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters())
        self.replay_buffer = ReplayBuffer(replay_buffer_size, transition=ACTransition)
        self.log_dict["loss/critic"] = 0
        self.log_dict["loss/actor"] = 0
        self.stop_update = False
        self.update_thread = Thread(target=self.update_runnable).start()

    def __del__(self):
        self.stop_update = True

    def _get_state_dict_(self):
        return {"model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()}

    def _load_impl_(self, state_dict):
        self.actor_critic.load_state_dict(state_dict["model_state_dict"])
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def get_correction_weight(self, p, q):
        m = torch.log((0.5 * (p + q)))
        jsd = 0.5 * (self.kld(m, torch.log(p)) + self.kld(m, torch.log(q)))
        return torch.exp(-jsd)

    def update_runnable(self):
        batch_generator = None
        while not self.stop_update:
            if len(self.replay_buffer) >= self.batch_size:
                if batch_generator is None:
                    batch_generator = self.replay_buffer.sample(self.batch_size)()
                try:
                    batch = next(batch_generator)
                except StopIteration:
                    batch_generator = None
                    continue
                state_batch = batch.state
                action_batch = batch.action
                next_state_batch = batch.next_state
                reward_batch = batch.reward
                exploration_prob_batch = batch.exploration_prob
                is_terminal = batch.is_terminal
                action_probs, v_current = self.actor_critic(state_batch)
                _, v_next = self.actor_critic(next_state_batch)
                advantage = reward_batch + (1. - is_terminal) * self.discount_factor * v_next - v_current
                # off-policy correction
                action_probs = action_probs.gather(-1, action_batch)
                weights = self.get_correction_weight(action_probs, exploration_prob_batch).detach()
                weights_sum = weights.sum()
                # loss calculation
                critic_loss = 0.5 * ((advantage.pow(2) * weights).sum() / weights_sum)
                actor_loss = torch.log(action_probs) * advantage.detach() * weights
                actor_loss = -actor_loss.sum() / weights_sum
                loss = actor_loss + critic_loss
                if not loss.isnan().any():
                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.actor_critic.parameters(), 100)
                    self.optimizer.step()
                    self.log_dict["loss/critic"] += critic_loss.item()
                    self.log_dict["loss/actor"] += actor_loss.item()
                else:
                    rospy.logwarn(f"NaNs in actor critic loss. Epoch {self.epoch}")
            else:
                time.sleep(1)

    def _update_impl(self, state_dict, reward):
        if self.train:
            if len(self.last_state_dict):
                self.replay_buffer.push(self.last_state_dict["state"], self.last_state_dict["robot_velocity"], self.action_index, state_dict["state"], reward, state_dict["is_terminal"],
                                        self.exploration_probs.gather(-1, self.action_index))
                # hindsight experience replay
                her_transition = self.create_her_transitions(self.last_state_dict, self.action_index, state_dict, reward, single_transition=True)
                if her_transition is not None:
                    her_state, her_velocity, her_action, _, _, _ = her_transition
                    self.actor_critic.eval()
                    with torch.no_grad():
                        her_exploration_probs = self.get_exploration_probs(self.actor_critic(her_state)[0], her_velocity).gather(-1, her_action)
                    self.actor_critic.train()
                    self.replay_buffer.push(*her_transition, her_exploration_probs)
        self.actor_critic.eval()
        with torch.no_grad():
            self.action_index = self.actor_critic(state_dict["state"].to(DEVICE))[0]
        self.actor_critic.train()


class SACQ(nn.Module):
    def __init__(self, state_dim, action_dim, layer_size):
        super(SACQ, self).__init__()
        self.dense1 = nn.Linear(state_dim + action_dim, layer_size)
        self.bnorm1 = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.bnorm2 = nn.BatchNorm1d(layer_size)
        self.out = nn.Linear(layer_size, 1)

    def forward(self, state, action):
        x = torch.relu(self.dense1(torch.cat([state, action], -1)))
        # x = self.bnorm1(x)
        x = torch.relu(self.dense2(x))
        # x = self.bnorm2(x)
        return self.out(x)


class SACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, layer_size, max_force):
        super(SACPolicy, self).__init__()
        self.dense1 = nn.Linear(state_dim, layer_size)
        self.bnorm1 = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(layer_size, layer_size)
        self.bnorm2 = nn.BatchNorm1d(layer_size)
        self.mean = nn.Linear(layer_size, action_dim)
        self.log_std = nn.Linear(layer_size, action_dim)
        self.log_std_min = -5
        self.log_std_max = 3
        self.max_force = max_force

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        # x = self.bnorm1(x)
        x = torch.relu(self.dense2(x))
        # x = self.bnorm2(x)
        log_std = torch.tanh(self.log_std(x))
        return self.mean(x), self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

    def get_action(self, state, sample=True):
        mean, log_std = self.forward(state)
        dist = torch.distributions.Normal(mean, log_std.exp())
        action = dist.rsample() if sample else mean
        action = torch.tanh(action)
        log_prob = dist.log_prob(action)
        log_prob = (log_prob - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob


class SACContext(ContinuesRLContext):
    def __init__(self,
                 critic_layer_size,
                 actor_layer_size,
                 replay_buffer_size,
                 tau,
                 **kwargs):
        super(SACContext, self).__init__(**kwargs)
        self.q1 = SACQ(self.state_dim, self.action_dim, critic_layer_size).to(DEVICE)
        self.q1_target = SACQ(self.state_dim, self.action_dim, critic_layer_size).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q1_optim = torch.optim.Adam(self.q1.parameters())
        self.q2 = SACQ(self.state_dim, self.action_dim, critic_layer_size).to(DEVICE)
        self.q2_target = SACQ(self.state_dim, self.action_dim, critic_layer_size).to(DEVICE)
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q2_optim = torch.optim.Adam(self.q2.parameters())
        self.actor = SACPolicy(self.state_dim, self.action_dim, actor_layer_size, self.max_force).to(DEVICE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters())
        self.log_alpha = torch.zeros(1, device=DEVICE, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha])
        self.tau = tau
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.actor_lock = Lock()
        self.stop_update = False
        self.update_thread = Thread(target=self.update_runnable).start()
        self.log_dict["loss/q1"] = 0
        self.log_dict["loss/q2"] = 0
        self.log_dict["loss/actor"] = 0
        self.log_dict["loss/alpha"] = 0
        self.max_norm = torch.linalg.vector_norm(torch.ones(self.action_dim))
        self.action_raw = None

    def _get_state_dict_(self):
        return {
            "q1": self.q1.state_dict(),
            "q1_optim": self.q1_optim.state_dict(),
            "q2": self.q2.state_dict(),
            "q2_optim": self.q2_optim.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def _load_impl_(self, state_dict):
        self.q1.load_state_dict(state_dict["q1"])
        self.q1_optim = torch.optim.Adam(self.q1.parameters())
        self.q1_optim.load_state_dict(state_dict["q1_optim"])
        self.q2.load_state_dict(state_dict["q2"])
        self.q2_optim = torch.optim.Adam(self.q2.parameters())
        self.q2_optim.load_state_dict(state_dict["q2_optim"])
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optim = torch.optim.Adam(self.actor.parameters())
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.log_alpha = torch.full_like(self.log_alpha, state_dict["log_alpha"], device=DEVICE, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha])
        self.alpha_optim.load_state_dict(state_dict["alpha_optim"])

    def _soft_copy(self, target_net, net):
        for target_p, p in zip(target_net.parameters(), net.parameters()):
            target_p.data.copy_(target_p.data * (1. - self.tau) + p.data * self.tau)

    def update_runnable(self):
        batch_generator = None
        while not self.stop_update:
            if len(self.replay_buffer) >= self.batch_size:
                if batch_generator is None:
                    batch_generator = self.replay_buffer.sample(self.batch_size)()
                try:
                    batch = next(batch_generator)
                except StopIteration:
                    batch_generator = None
                    continue
                state_batch = batch.state
                action_batch = batch.action
                next_state_batch = batch.next_state
                reward_batch = batch.reward
                is_terminal_batch = batch.is_terminal
                print(reward_batch[is_terminal_batch.bool().logical_not()])
                alpha = self.log_alpha.exp()
                self.log_dict["loss/alpha"] += alpha.item()
                with torch.no_grad():
                    actor_next_state_actions, actor_next_state_actions_log_probs = self.actor.get_action(next_state_batch)
                    min_q = torch.min(self.q1_target(next_state_batch, actor_next_state_actions), self.q2_target(next_state_batch, actor_next_state_actions))
                    q_target = reward_batch.flatten() + self.discount_factor * (1 - is_terminal_batch.flatten()) * (min_q - alpha * actor_next_state_actions_log_probs).view(-1)
                q1_loss = torch.nn.MSELoss()(self.q1(state_batch, action_batch).view(-1), q_target)
                q2_loss = torch.nn.MSELoss()(self.q2(state_batch, action_batch).view(-1), q_target)
                with self.actor_lock:
                    actor_state_actions, actor_state_actions_log_probs = self.actor.get_action(state_batch)
                    actor_loss = (alpha * actor_state_actions_log_probs - torch.min(self.q1(state_batch, actor_state_actions), self.q2(state_batch, actor_state_actions)).view(-1)).mean()
                    if not actor_loss.isnan().any():
                        self.actor_optim.zero_grad()
                        actor_loss.backward()
                        self.actor_optim.step()
                        self.log_dict["loss/actor"] += actor_loss.item()
                    else:
                        rospy.logwarn(f"NaNs in actor loss. Epoch {self.epoch}")
                if not q1_loss.isnan().any():
                    self.q1_optim.zero_grad()
                    q1_loss.backward()
                    self.q1_optim.step()
                    self._soft_copy(self.q1_target, self.q1)
                    self.log_dict["loss/q1"] += q1_loss.item()
                else:
                    rospy.logwarn(f"NaNs in q1 loss. Epoch {self.epoch}")
                if not q2_loss.isnan().any():
                    self.q2_optim.zero_grad()
                    q2_loss.backward()
                    self.q2_optim.step()
                    self._soft_copy(self.q2_target, self.q2)
                    self.log_dict["loss/q2"] += q2_loss.item()
                else:
                    rospy.logwarn(f"NaNs in q2 loss. Epoch {self.epoch}")
                alpha_loss = (-self.log_alpha * (actor_state_actions_log_probs - self.action_dim).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
            else:
                time.sleep(1)

    def _update_impl(self, state_dict, reward):
        if self.train:
            if len(self.last_state_dict):
                self.replay_buffer.push(self.last_state_dict["state"], self.last_state_dict["robot_velocity"], self.action_raw, state_dict["state"], reward, state_dict["is_terminal"])
                # hindsight experience replay
                her_transition = self.create_her_transitions(self.last_state_dict, self.action_raw, state_dict, reward, single_transition=True)
                if her_transition is not None:
                    self.replay_buffer.push(*her_transition)
        with self.actor_lock:
            self.actor.eval()
            with torch.no_grad():
                self.action_raw, _ = self.actor.get_action(state_dict["state"], sample=self.train)
                self.action = self.action_raw / self.max_norm * self.max_force
            self.actor.train()
