import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.td3 import td3Critic
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F


class td3Learner_noise1:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        print("self.n_agents:",self.n_agents)
        self.n_actions = args.n_actions
        self.logger = logger
        self.policy_freq = args.learn_interval

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = td3Critic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.actors_set = set()
        self.actors = []
        self.actor_targets = []
        self.actor_optimisers = []


        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, nr_of_steps):

        total_it = 0
        # print("batch",batch) Batch Size:100 Max_seq_len:2 Keys:dict_keys(['state', 'obs', 'actions', 'avail_actions', 'reward', 'terminated', 'filled']) Groups:dict_keys(['agents'])
        for _ in range(nr_of_steps):
            # Get the relevant quantities
            rewards = batch["reward"][:, :-1]
            # print("rewards:",rewards.shape)  rewards: torch.Size([100, 1, 1])
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask1 = batch["filled"][:, :-1].float()
            mask1[:, 1:] = mask1[:, 1:] * (1 - terminated[:, :-1])
            mask2 = batch["filled"][:, :-1].float()
            mask2[:, 1:] = mask2[:, 1:] * (1 - terminated[:, :-1])

            total_it += 1
            # Train the critic batched
            with th.no_grad():
                target_actions = []
                self.target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    agent_target_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, test_mode=True,
                                                                       critic=self.target_critic, target_mac=True)
                    target_actions.append(agent_target_outs)
                target_actions = th.stack(target_actions, dim=1)  # Concat over time
                target_vals = []
                for t in range(1, batch.max_seq_length):
                    target_inputs = self._build_inputs(batch, t=t)
                    target_critic_out1, target_critic_out2, _ = self.target_critic(target_inputs,
                                                                                   target_actions[:, t:t + 1].detach())
                    target_critic_out1 = target_critic_out1.view(batch.batch_size, -1, 1)
                    target_critic_out2 = target_critic_out2.view(batch.batch_size, -1, 1)
                    # print("target_critic_out2.shape",target_critic_out2.shape)
                    # print("\n")
                    # print("target_critic_out2:",target_critic_out2)
                    target_critic_out = th.min(target_critic_out1, target_critic_out2)
                    # print("target_critic_out.shape:", target_critic_out.shape)  torch.Size([100, 1, 1])
                    # print("target_critic_out:",target_critic_out)
                    target_vals.append(target_critic_out)
                    # print("target_vals:",target_vals)
                target_vals = th.stack(target_vals, dim=1)

            q_taken1 = []
            q_taken2 = []
            for t in range(batch.max_seq_length - 1):
                inputs = self._build_inputs(batch, t=t)
                critic_out1, critic_out2, _ = self.critic(inputs, actions[:, t:t+1].detach())
                critic_out1 = critic_out1.view(batch.batch_size, -1, 1)
                critic_out2 = critic_out2.view(batch.batch_size, -1, 1)
                q_taken1.append(critic_out1)
                q_taken2.append(critic_out2)
            q_taken1 = th.stack(q_taken1, dim=1)
            q_taken2 = th.stack(q_taken2, dim=1)

            q_taken1 = q_taken1.view(batch.batch_size, -1, 1)
            q_taken2 = q_taken2.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
            targets = rewards.expand_as(target_vals) + self.args.gamma * (1 - terminated.expand_as(target_vals))*target_vals

            td_error1 = (q_taken1 - targets.detach())
            td_error2 = (q_taken2 - targets.detach())
            mask1 = mask1.expand_as(td_error1)
            mask2 = mask2.expand_as(td_error2)
            masked_td_error1 = td_error1 * mask1
            masked_td_error2 = td_error2 * mask2
            loss = (masked_td_error1 ** 2).sum() / mask1.sum() + (masked_td_error2 ** 2).sum() / mask2.sum()

            # Optimize the critic
            self.critic_optimiser.zero_grad()
            loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()

            # Delayed policy updates
            if total_it % self.policy_freq == 0:
                mac_out = []
                chosen_action_qvals = []
                self.mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length - 1):
                    agent_outs = self.mac.forward(batch, t=t, select_actions=True)["actions"].view(batch.batch_size,
                                                                                                   self.n_agents,
                                                                                                   self.n_actions)

                    for idx in range(self.n_agents):
                        tem_joint_act = actions[:, t:t+1].detach().clone().view(batch.batch_size, -1, self.n_actions)
                        tem_joint_act[:, idx] = agent_outs[:, idx]
                        q, _ = self.critic.Q1(self._build_inputs(batch, t=t), tem_joint_act)
                        chosen_action_qvals.append(q.view(batch.batch_size, -1, 1))
                        actors_set_divi = set()
                        actors_divi = []
                        actor_targets = []
                        actor_optimisers = []

                    mac_out.append(agent_outs)
                mac_out = th.stack(mac_out, dim=1)
                chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
                pi = mac_out

                # Compute the actor loss
                pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

                # Optimise agents
                self.agent_optimiser.zero_grad()
                pg_loss.backward()
                agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
                self.agent_optimiser.step()


                if getattr(self.args, "target_update_mode", "hard") == "hard":
                    self._update_targets()
                elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
                    self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
                else:
                    raise Exception(
                        "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

                if t_env - self.log_stats_t >= self.args.learner_log_interval:
                    self.logger.log_stat("critic_loss", loss.item(), t_env)
                    self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
                    mask_elems1 = mask1.sum().item()
                    mask_elems2 = mask2.sum().item()
                    self.logger.log_stat("td_error_abs", masked_td_error1.abs().sum().item() / mask_elems1 + masked_td_error2.abs().sum().item() / mask_elems2, t_env)

                    self.logger.log_stat("q_taken_mean", (q_taken1 * mask1).sum().item() / mask_elems1, t_env)
                    self.logger.log_stat("target_mean", targets.sum().item() / mask_elems1, t_env)
                    self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
                    self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
                    self.log_stats_t = t_env

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        # The centralized critic takes the state input, not observation
        inputs.append(batch["state"][:, t])

        if self.args.recurrent_critic:
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t - 1])

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))