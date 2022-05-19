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


class NaturalNext(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.obs_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, obs_dim))
        self.apply(utils.weight_init)
    
    def forward(self, obs):
        obs_pred = self.obs_pred_net(obs)
        return obs_pred


class DIAYNAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, natural_next_scale, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder

        self.natural_next_scale = natural_next_scale

        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        # create diayn
        self.diayn_diff = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        self.natural_next = NaturalNext(self.obs_dim - self.skill_dim,
                                        kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.diayn_diff_criterion = nn.CrossEntropyLoss()
        self.natural_next_criterion = nn.MSELoss()

        # optimizers
        self.diayn_diff_opt = torch.optim.Adam(self.diayn_diff.parameters(), lr=self.lr)
        self.natural_next_opt = torch.optim.Adam(self.natural_next.parameters(), lr=self.lr)

        # normalizer
        self.normalizer = None
        self.normalizer_exp = 0.9

        self.diayn_diff.train()
        self.natural_next.train()

    def update_normalizer(self, obs_max):
        with torch.no_grad():
            if self.normalizer is None:
                self.normalizer = obs_max
            else:
                self.normalizer = self.normalizer_exp*self.normalizer + (1 - self.normalizer_exp)*obs_max

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

    def init_extra_meta(self):
        skill = np.ones(self.skill_dim, dtype=np.float32)
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

    def update_natural_next(self, obs_curr, obs_diff, step):
        metrics = dict()

        loss = self.compute_natural_next_loss(obs_curr, obs_diff)

        self.natural_next_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.natural_next_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['natural_next_loss'] = loss.item()

        return metrics

    # TODO: use (rew_nextobs - rew_currobs) for reward_diayn_diff?
    def compute_intr_reward(self, skill, obs_curr, obs_diff, step):
        z_hat = torch.argmax(skill, dim=1)

        # diayn diff
        d_pred = self.diayn_diff(obs_diff)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward_diayn_diff = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward_diayn_diff = reward_diayn_diff.reshape(-1, 1)

        # natural next
        # tmp_normalizer = torch.as_tensor([
        #     1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1,
        #     5, 3, 15, 30, 29,
        #     41, 31, 30, 39
        # ], dtype=torch.float32, device=self.device)
        
        obs_curr, obs_diff = obs_curr / self.normalizer, obs_diff / self.normalizer
        
        d_pred = self.natural_next(obs_curr)
        reward_natural_next = torch.sqrt(torch.sum((d_pred - obs_diff) ** 2, axis=1) / (self.obs_dim - self.skill_dim))
        reward_natural_next = reward_natural_next.reshape(-1, 1)

        # total reward
        reward = reward_diayn_diff + reward_natural_next

        return reward_diayn_diff*self.diayn_scale, reward_natural_next*self.natural_next_scale

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

    def compute_natural_next_loss(self, obs, obs_diff):
        """
        DF Loss
        """
        
        obs, obs_diff = obs / self.normalizer, obs_diff / self.normalizer

        d_pred = self.natural_next(obs)
        d_loss = self.natural_next_criterion(d_pred, obs_diff)

        return d_loss

    def update(self, replay_iter, extra_replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        batch_extra = next(extra_replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        obs_extra, action_extra, extr_reward_extra, discount_extra, next_obs_extra, skill_extra = utils.to_torch(
            batch_extra, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        obs_extra = self.aug_and_encode(obs_extra)
        next_obs_extra = self.aug_and_encode(next_obs_extra)

        obs_max, _ = torch.max(torch.abs(obs), axis=0)
        self.update_normalizer(obs_max)

        obs_diff = next_obs - obs
        obs_diff_extra = next_obs_extra - obs_extra

        if self.reward_free:
            metrics.update(self.update_diayn_diff(skill, obs_diff, step))
            metrics.update(self.update_natural_next(obs_extra, obs_diff_extra, step))

            with torch.no_grad():
                intr_reward_diayn_diff, intr_reward_natrual_next = self.compute_intr_reward(skill, obs, obs_diff, step)
                intr_reward = intr_reward_diayn_diff + intr_reward_natrual_next

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
