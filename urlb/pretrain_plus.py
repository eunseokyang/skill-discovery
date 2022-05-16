import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = PRIMAL_TASKS[self.cfg.domain]
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        self.extra_replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                        self.work_dir / 'extra_buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)

        self.extra_replay_loader = make_replay_loader(self.extra_replay_storage,
                                                      cfg.extra_replay_buffer_size,
                                                      cfg.batch_size,
                                                      cfg.extra_replay_buffer_num_workers,
                                                      False, cfg.extra_nstep, cfg.discount)
                                        
        self._replay_iter = None
        self._extra_replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def extra_replay_iter(self):
        if self._extra_replay_iter is None:
            self._extra_replay_iter = iter(self.extra_replay_loader)
        return self._extra_replay_iter

    def save_tracking_csv(self, arr, dirname, fname, fmt='%.18e'):
        save_txt_dir = (self.work_dir / dirname)
        save_txt_dir.mkdir(exist_ok=True)
        np.savetxt(save_txt_dir / f'{self.global_frame}_{fname}.csv', arr, fmt=fmt, delimiter=',')

    def save_tracking_np(self, arr, dirname, fname):
        save_txt_dir = (self.work_dir / dirname)
        save_txt_dir.mkdir(exist_ok=True)
        np.save(save_txt_dir / f'{self.global_frame}_{fname}.npy', arr)

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        ## intrinsic reward
        intrinsic_reward_list = []
        intrinsic_total_reward = 0

        observation_list = []

        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        skill_num = np.argmax(meta['skill'])
        while eval_until_episode(episode):
            ## for tracking intrinsic rewards
            observations, skills = [], []

            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                observations.append(time_step.observation)
                skills.append(meta['skill'])
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            observation_list.append(observations)

            obs = torch.as_tensor(np.array(observations))
            obs_diff = (obs[1:, :] - obs[:-1, :]).to(self.device)
            obs = obs[:-1, :].to(self.device)
            skills = torch.as_tensor(np.array(skills)[1:, :], device=self.device)
            with torch.no_grad():
                intr_reward_diayn_diff, intr_reward_natural_next = self.agent.compute_intr_reward(skills, obs, obs_diff, None)
            intr_reward_diayn_diff = intr_reward_diayn_diff.detach().cpu().squeeze(1).numpy()
            intr_reward_natural_next = intr_reward_natural_next.detach().cpu().squeeze(1).numpy()
            intrinsic_reward_list.append(intr_reward_diayn_diff)
            intrinsic_reward_list.append(intr_reward_natural_next)

            episode += 1
            self.video_recorder.save(f'{self.global_frame}_{skill_num}.mp4')

        intrinsic_reward_list = np.array(intrinsic_reward_list)
        intrinsic_total_reward = np.sum(intrinsic_reward_list)

        # (trajectory_len, num_eval_episode)
        self.save_tracking_csv(intrinsic_reward_list.T, dirname='intr_reward', fname='', fmt='%2.4f')
        # (trajectory_len, num_eval_episode, obs_dim)
        self.save_tracking_np(np.array(observation_list), dirname='obs', fname='')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_intrinsic_reward', intrinsic_total_reward / episode)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        extra_every_step = utils.Every(self.cfg.extra_every_frames,
                                       self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        extra_meta = self.agent.init_extra_meta()

        self.extra_mode = False
        meta_use = meta

        self.replay_storage.add(time_step, meta, False)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('extra_buffer_size', len(self.extra_replay_storage))
                        log('step', self.global_step)
                
                ## turn back to normal mode
                if self.extra_mode:
                    self.extra_mode = False
                    meta_use = self.agent.init_meta()

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step, meta_use, False)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            if extra_every_step(self.global_step + self.cfg.extra_train_step):
                self.replay_storage.add(time_step, meta_use, True)

                self.extra_mode = True
                meta_use = extra_meta

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
            
            if self.extra_mode:
                ## TODO: update extra skills?
                "pass"
            else:
                ## TODO: how to update skills?
                meta_use = self.agent.update_meta(meta_use, self.global_step, time_step)

            # sample action
            if self.extra_mode:
                # action = np.random.uniform(self.train_env.action_spec().minimum,
                #                            self.train_env.action_spec().maximum,
                #                            size=self.train_env.action_spec().shape
                #                            ).astype(np.float32)
                action = np.zeros(self.train_env.action_spec().shape).astype(np.float32)
            else:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta_use,
                                            self.global_step,
                                            eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.extra_replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            if self.extra_mode:
                self.extra_replay_storage.add(time_step, meta_use, False)
            else:
                self.replay_storage.add(time_step, meta_use, False)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)


@hydra.main(config_path='.', config_name='pretrain_plus')
def main(cfg):
    from pretrain_plus import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
