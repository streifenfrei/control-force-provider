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
        x = torch.nn.functional.mish(self.dense1(state))
        # x = self.bnorm1(x)
        x = torch.nn.functional.mish(self.dense2(x))
        # x = self.bnorm2(x)
        return self.softmax(10 * torch.tanh(self.actor(x))), self.critic(x)


class A2CContext(DiscreteRLContext):
    def __init__(self,
                 layer_size,
                 entropy_beta,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor_critic = ActorCritic(self.state_dim, len(self.action_space), layer_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters())
        self.update_interval = 2 * self.max_episode_length
        self.update_step_size = int(self.batch_size / self.update_interval)
        self.update_steps = math.ceil(self.robot_batch / self.update_step_size)
        assert self.update_step_size > 0
        self.state_stack = []
        self.actions_stack = []
        self.rewards_stack = []
        self.is_terminal_stack = []
        self.reached_goal_stack = []
        self.entropy_beta = float(entropy_beta)
        self.weight_limit = torch.tensor(0.1, device=DEVICE)
        self.log_dict["loss/critic"] = 0
        self.log_dict["loss/actor"] = 0
        self.log_dict["loss/entropy"] = 0
        self.explore = False

    def __del__(self):
        self.stop_update = True

    def _get_state_dict_(self):
        return {"model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()}

    def _load_impl_(self, state_dict):
        self.actor_critic.load_state_dict(state_dict["model_state_dict"])
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def _update_impl(self, state_dict, reward):
        if self.train:
            if len(self.last_state_dict):
                self.state_stack.append(state_dict["state"].cpu())
                self.actions_stack.append(self.action_index.cpu())
                self.rewards_stack.append(reward.cpu())
                self.is_terminal_stack.append(state_dict["is_terminal"].cpu())
                self.reached_goal_stack.append(state_dict["reached_goal"].cpu())
            if len(self.state_stack) == self.update_interval:
                all_states = torch.stack(self.state_stack)
                all_actions = torch.stack(self.actions_stack)
                all_rewards = torch.stack(self.rewards_stack)
                all_terminals = torch.stack(self.is_terminal_stack)
                all_reached_goals = torch.stack(self.reached_goal_stack)
                loss_critic_log = 0
                loss_actor_log = 0
                loss_entropy_log = 0
                for batch_start in range(0, self.robot_batch, self.update_step_size):
                    batch_end = min(batch_start + self.update_step_size, self.robot_batch)
                    states = all_states[:, batch_start:batch_end, :].to(DEVICE)
                    actions = all_actions[:, batch_start:batch_end, :].to(DEVICE)
                    rewards = all_rewards[:, batch_start:batch_end, :].to(DEVICE)
                    terminals = all_terminals[:, batch_start:batch_end, :].to(DEVICE)
                    reached_goal = all_reached_goals[:, batch_start:batch_end, :].to(device=DEVICE, dtype=torch.float)
                    no_terminals = terminals.logical_not()
                    terminals_float = terminals.float()
                    returns = torch.zeros_like(rewards)
                    is_valid = torch.zeros_like(terminals)
                    for i in reversed(range(self.update_interval)):
                        i_plus_one = min(i + 1, self.update_interval - 1)
                        i_minus_one = max(i - 1, 0)
                        returns[i, :, :] = (1 - terminals_float[i, :, :]) * self.discount_factor * returns[i_plus_one, :, :] + rewards[i, :, :]
                        is_valid[i, :, :] = (is_valid[i_plus_one, :, :].logical_or(terminals[i, :, :])).logical_and(no_terminals[i, :, :].logical_or(no_terminals[i_minus_one, :, :]))
                        reached_goal[i, :, :] = reached_goal[i, :, :].logical_or(reached_goal[i_plus_one].logical_and(no_terminals[i, :, :]))
                    states = torch.masked_select(states, is_valid).reshape([-1, states.size(-1)])
                    actions = torch.masked_select(actions, is_valid).reshape([-1, actions.size(-1)])
                    returns = torch.masked_select(returns, is_valid).reshape([-1, returns.size(-1)])
                    reached_goal = torch.masked_select(reached_goal, is_valid).reshape([-1, reached_goal.size(-1)])
                    action_probs, value = self.actor_critic(states)
                    entropy = -(action_probs * torch.log(action_probs)).sum(-1)
                    action_probs = action_probs.gather(-1, actions)
                    advantage = returns - value
                    success_ratio = torch.min(reached_goal.sum() / reached_goal.numel(), self.weight_limit)
                    fail_ratio = 1 - success_ratio
                    weights = reached_goal * (fail_ratio / (success_ratio + EPSILON)) + (1 - reached_goal)
                    # loss calculation
                    actor_loss = -((weights * torch.log(action_probs) * advantage.detach())).mean()
                    critic_loss = (weights * advantage.pow(2)).mean()
                    entropy_loss = ((weights - 1) * entropy).mean()
                    loss = actor_loss + critic_loss + self.entropy_beta * entropy_loss
                    if not loss.isnan().any():
                        self.optimizer.zero_grad()
                        loss.backward()
                        clip_grad_norm_(self.actor_critic.parameters(), 1)
                        self.optimizer.step()
                        loss_actor_log += actor_loss.item()
                        loss_critic_log += critic_loss.item()
                        loss_entropy_log += entropy.mean().item()
                    else:
                        rospy.logwarn(f"NaNs in actor critic loss. Epoch {self.epoch}")
                self.log_dict["loss/critic"] += loss_critic_log / self.update_steps
                self.log_dict["loss/actor"] += loss_actor_log / self.update_steps
                self.log_dict["loss/entropy"] += loss_entropy_log / self.update_steps
                self.state_stack = []
                self.actions_stack = []
                self.rewards_stack = []
                self.is_terminal_stack = []
                self.reached_goal_stack = []
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
        self.explore = False

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
                was_not_terminal = self.last_state_dict["is_terminal"].logical_not()
                self.replay_buffer.push(torch.masked_select(self.last_state_dict["state"], was_not_terminal).reshape([-1, self.state_dim]),
                                        torch.masked_select(self.last_state_dict["robot_velocity"], was_not_terminal).reshape([-1, 3]),
                                        torch.masked_select(self.action_raw, was_not_terminal).reshape([-1, 3]),
                                        torch.masked_select(state_dict["state"], was_not_terminal).reshape([-1, self.state_dim]),
                                        torch.masked_select(reward, was_not_terminal).reshape([-1, 1]),
                                        torch.masked_select(state_dict["is_terminal"], was_not_terminal).reshape([-1, 1]))
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
