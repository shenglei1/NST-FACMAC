import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.facmac import FACMACCritic
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_ablations import VDNState, QMixerNonmonotonic
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
import numpy as np


class FACMACLearnerSoft:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = FACMACCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1: # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn-s":
                self.mixer = VDNState(args)
            elif args.mixer == "qmix-nonmonotonic":
                self.mixer = QMixerNonmonotonic(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha,
                                           eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr,
                                        eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                            eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr,
                                         eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))



        self.log_stats_t = -self.args.learner_log_interval - 1

    def softmax_operator(self, q_vals, noise_pdf=None):
        print("q_vals",q_vals)
        max_q_vals = th.max(q_vals, 1, keepdim=True).values
        print("max_q_vals",max_q_vals)
        max_q_vals_indx = th.max(q_vals, 1, keepdim=True)[1]
        print("max_q_vals",max_q_vals.shape)  #[bs, 1, n_agents]
        print("max_q_vals_indx",max_q_vals_indx)


        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = th.exp(0.05 * norm_q_vals)
        Q_mult_e = q_vals * e_beta_normQ

        numerators = Q_mult_e
        denominators = e_beta_normQ


        numerators /= noise_pdf.to(device)
        denominators /= noise_pdf.to(device)

        sum_numerators = th.sum(numerators, 1)
        sum_denominators = th.sum(denominators, 1)

        softmax_q_vals = sum_numerators / sum_denominators

        softmax_q_vals = th.unsqueeze(softmax_q_vals, 1)
        return softmax_q_vals

    def calc_pdf(self, samples, mu=0):
        pdfs = 1 / (self.args.act_noise * np.sqrt(2 * np.pi)) * th.exp(
            - (samples - mu) ** 2 / (2 * self.args.act_noise ** 2))
        pdf = th.prod(pdfs, dim=3)  #返回输入张量给定维度上每行的积。
        # print("pdf",pdf.shape)     #[10, 2, 4]
        # print("pdfs",pdfs.shape)  #[10, k, 4, 2]
        return pdf  #[batch_size, k, n_agents]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        actions = batch["actions"][:, :-1]  #[batch_size,1,n_agents,n-actions]
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # print("mask[:, 1:]_size",mask[:, 1:].shape)  [batch_size, 0, 1]

        # Train the critic batched
        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        # print("batch.batch_size",batch.batch_size)  100

        for t in range(batch.max_seq_length):
            action_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, test_mode=True,
                                                         critic=self.target_critic, target_mac=True)  #batch_size, self.n_agents, self.args.n_actions


            target_actions.append(action_outs)
        # print("target_actions",target_actions)  #list[tensor[100, n_agent, n_action]
        target_actions = th.stack(target_actions, dim=1)  # Concat over time [100, 2, n_agent, n_action]
        noise = th.randn(batch.batch_size, self.args.sample, target_actions.shape[2], target_actions.shape[3])
        noise = noise * self.args.act_noise
        noise_pdf = self.calc_pdf(noise)  # [batch_size, k, n_agents]
        noise = noise.clamp(-0.25, 0.25)


        target_qvals = []
        target_qvals_ind = []
        self.target_critic.init_hidden(batch.batch_size)
        for t in range(1, batch.max_seq_length):
            target_inputs = self._build_inputs(batch, t=t)

            target_actions_sample = target_actions[:, t:t + 1].repeat((1, self.args.sample, 1,1))
            target_actions_sample_noise = noise + target_actions_sample
            for i in range(self.args.sample):
                target_critic_out, \
                self.target_critic.hidden_states = self.target_critic(target_inputs, target_actions_sample_noise[:,i:i+1,:,:].detach(),
                                                                      self.target_critic.hidden_states)
                target_qvals_ind.append(target_critic_out.view(batch.batch_size, -1, 1))

        target_qvals_ind = th.stack(target_qvals_ind,dim=1)
        target_qvals_ind = th.squeeze(target_qvals_ind,3) # [bs,k,n_agents(的q)]
        print("target_qvals_ind",target_qvals_ind.shape)
        softmax_Q_ind = self.softmax_operator(target_qvals_ind, noise_pdf)
        print("softmax_Q_ind",softmax_Q_ind.shape)  #[10, 1, n_agents]

        if self.mixer is not None:
            target_critic_out = self.target_mixer(softmax_Q_ind.view(batch.batch_size, -1, 1),
                                                  batch["state"][:,1:2])
        #   print("target_critic_out_shape", target_critic_out.shape)  # [bs, 1, 1]
            target_qvals.append(target_critic_out)
        target_qvals = th.stack(target_qvals, dim=1)  # Concat over time  [100, 1, 1, 1_q]
        print("target_qvals",target_qvals.shape)


        chosen_action_qvals = []
        mac_out_critic = []
        self.critic.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            inputs = self._build_inputs(batch, t=t)
            critic_out, self.critic.hidden_states = self.critic(inputs.to(device),
                                                                actions[:, t:t + 1].detach().to(device),
                                                                self.critic.hidden_states)

            mac_out_critic.append(critic_out.view(batch.batch_size, -1, 1))
            if self.mixer is not None:
                critic_out = self.mixer(critic_out.view(batch.batch_size, -1, 1), batch["state"][:, t:t + 1])
            chosen_action_qvals.append(critic_out)

        mac_out_critic = th.stack(mac_out_critic, dim=1)  # Concat over time [100, 1, 2, 1]
        chosen_action_qvals1 = th.stack(chosen_action_qvals, dim=1)



        if self.mixer is not None:
            chosen_action_qvals1 = chosen_action_qvals1.view(batch.batch_size, -1, 1)
            target_qvals = target_qvals.view(batch.batch_size, -1, 1)
        else:
            chosen_action_qvals1 = chosen_action_qvals1.view(batch.batch_size, -1, self.n_agents)
            target_qvals = target_qvals.view(batch.batch_size, -1, self.n_agents)

        targets = rewards.expand_as(target_qvals) + self.args.gamma * (
                    1 - terminated.expand_as(target_qvals)) * target_qvals
        td_error = (targets.detach() - chosen_action_qvals1)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # DEBUG
        if getattr(self.args, "plot_loss_network", False):
            from torchviz import make_dot
            dot = make_dot(loss, params=self.named_params)
            dot.format = 'svg'
            dot.render()


        mac_out = []
        chosen_action_qvals_ = []
        self.mac.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)
        # print("batch.max_seq_length - 1",batch.max_seq_length - 1)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t, select_actions=True)["actions"].view(batch.batch_size,
                                                                                           self.n_agents,
                                                                                           self.n_actions)
            print("agent_outs", agent_outs)  # [100,2,3]
            q, self.critic.hidden_states = self.critic(self._build_inputs(batch, t=t), agent_outs,
                                                       self.critic.hidden_states)
            if self.mixer is not None:
                q = self.mixer(q.view(batch.batch_size, -1, 1), batch["state"][:, t:t + 1])
            chosen_action_qvals_.append(q.view(batch.batch_size, -1, 1))
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        print("mac_out", mac_out.shape)  # [100, 1, 2, 3]
        # print(mac_out)
        chosen_action_qvals_ = th.stack(chosen_action_qvals_, dim=1)
        pi = mac_out
        print("chosen_action_qvals", chosen_action_qvals_.shape)  # [100, 1, 1, 1]

        # Compute the actor loss
        pg_loss = -chosen_action_qvals_.mean() + (pi ** 2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = episode_num
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau = getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception("unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.log_stats_t = t_env


    def _update_targets_soft(self, tau):

        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated target network (soft update tau={})".format(tau))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        if self.args.recurrent_critic:
            # The individual Q conditions on the global action-observation history and individual action
            inputs.append(batch["obs"][:, t].repeat(1, self.args.n_agents, 1).view(bs, self.args.n_agents, -1))
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t].repeat(1, self.args.n_agents, 1).
                                                view(bs, self.args.n_agents, -1)))
                else:
                    inputs.append(batch["actions"][:, t - 1].repeat(1, self.args.n_agents, 1).
                                  view(bs, self.args.n_agents, -1))
        else:
            print("batch[]",batch["obs"].shape)  #[10, 2, 4, 4]
            print("batch[][:, t]",batch["obs"][:, t].shape)  #[10, 4, 4]
            inputs.append(batch["obs"][:, t])

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))