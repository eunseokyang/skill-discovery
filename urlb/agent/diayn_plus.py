import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent


class DIAYN(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, obs):
        skill_pred = self.skill_pred_net(obs)
        return skill_pred


class DIAYNAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        # create diayn
        self.diayn = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        self.diayn_diff = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()
        self.diayn_diff_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)
        self.diayn_diff_opt = torch.optim.Adam(self.diayn_diff.parameters(), lr=self.lr)

        self.diayn.train()
        self.diayn_diff.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self, i=-1):
        if i > -1:
            return self.choose_meta(i)
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def choose_meta(self, i):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[i] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, skill=-1): 
        if global_step % self.update_skill_every_step == 0:
            if skill > -1:
                return self.choose_meta(skill)
            return self.init_meta()
        return meta

    def update_diayn(self, skill, curr_obs, step):
        metrics = dict()

        loss, df_accuracy = self.compute_diayn_loss(curr_obs, skill)

        self.diayn_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.diayn_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            metrics['diayn_acc'] = df_accuracy

        return metrics

    def update_diayn_diff(self, skill, obs_diff, step):
        metrics = dict()

        loss, df_accuracy = self.compute_diayn_diff_loss(obs_diff, skill)

        self.diayn_diff_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.diayn_diff_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_diff_loss'] = loss.item()
            metrics['diayn_diff_acc'] = df_accuracy

        return metrics

    # TODO: use (rew_nextobs - rew_currobs) for reward_diayn_diff?
    def compute_intr_reward(self, skill, next_obs, obs_diff, step):
        z_hat = torch.argmax(skill, dim=1)

        # diayn
        d_pred = self.diayn(next_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        # targ = torch.full(d_pred_log_softmax.shape, 1 / self.skill_dim).to(self.device)
        # reward_diayn = -F.kl_div(d_pred_log_softmax, targ, reduction='none').sum(axis=1)
        reward_diayn = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)  
        reward_diayn = reward_diayn.reshape(-1, 1)

        # diayn diff
        d_pred = self.diayn_diff(obs_diff)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward_diayn_diff = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward_diayn_diff = reward_diayn_diff.reshape(-1, 1)

        # total reward
        reward = reward_diayn + reward_diayn_diff

        return reward_diayn*self.diayn_scale, reward_diayn_diff*self.diayn_scale

    def compute_diayn_loss(self, curr_state, skill):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn(curr_state)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy

    def compute_diayn_diff_loss(self, state_diff, skill):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn_diff(state_diff)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_diff_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        obs_diff = next_obs - obs

        if self.reward_free:
            metrics.update(self.update_diayn(skill, obs, step))
            metrics.update(self.update_diayn_diff(skill, obs_diff, step))

            with torch.no_grad():
                intr_reward_diayn, intr_reward_diayn_diff = self.compute_intr_reward(skill, next_obs, obs_diff, step)
                intr_reward = intr_reward_diayn + intr_reward_diayn_diff

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
