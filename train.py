import os
import time
import dmc2gym
import hydra
import torch
import numpy as np
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
import wandb
import random
from hydra.utils import get_original_cwd

torch.backends.cudnn.benchmark = True

def make_env(cfg, test=False):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    if test:
        env = dmc2gym.make(domain_name=domain_name,
                           task_name=task_name,
                           test=test,
                           seed=cfg.seed,
                           difficulty=None if cfg.test_difficulty == "None" else cfg.test_difficulty,
                           dynamic=cfg.test_dynamic,
                           background_dataset_path=cfg.test_background_dataset_path,
                           background_dataset_videos=cfg.test_background_dataset_videos,
                           background_kwargs=None if cfg.test_background_kwargs == "None" else cfg.test_background_kwargs,
                           camera_kwargs=None if cfg.test_camera_kwargs == "None" else cfg.test_camera_kwargs,
                           color_kwargs=None if cfg.test_color_kwargs == "None" else cfg.test_color_kwargs,
                           visualize_reward=False,
                           from_pixels=True,
                           height=cfg.image_size,
                           width=cfg.image_size,
                           frame_skip=cfg.action_repeat,
                           camera_id=camera_id)

    else:
        env = dmc2gym.make(domain_name=domain_name,
                           task_name=task_name,
                           test=test,
                           seed=cfg.seed,
                           difficulty=None if cfg.difficulty == "None" else cfg.difficulty,
                           dynamic=cfg.dynamic,
                           background_dataset_path=cfg.background_dataset_path,
                           background_dataset_videos=cfg.background_dataset_videos,
                           background_kwargs=None if cfg.background_kwargs == "None" else cfg.background_kwargs,
                           camera_kwargs=None if cfg.camera_kwargs == "None" else cfg.camera_kwargs,
                           color_kwargs=None if cfg.color_kwargs == "None" else cfg.color_kwargs,
                           visualize_reward=False,
                           from_pixels=True,
                           height=cfg.image_size,
                           width=cfg.image_size,
                           frame_skip=cfg.action_repeat,
                           camera_id=camera_id)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.save_dir = os.path.join(self.work_dir, "trained_models")

        self.cfg = cfg

        wandb.init(project="ted", group=f"{cfg.wandb_group_name}", job_type="train",
                   config={
                        "env_name": cfg.env,
                        "exp_type": cfg.wandb_exp_type,
                        "agent_type": cfg.agent_type,
                        "learning_rate": cfg.lr,
                        "encoder learning rate": cfg.encoder_lr,
                        "action_repeat": cfg.action_repeat,
                        "num_train_steps": cfg.num_train_steps,
                        "num_test_steps": cfg.num_test_steps,
                        "batch_size": 128,
                        "seed": cfg.seed,
                        "ted": cfg.ted,
                        "ted_coef": cfg.ted_coef
                    })

        #wandb.init(config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.train_env = make_env(cfg, test=False)
        self.env = self.train_env

        self.test_env = make_env(cfg, test=True)

        if cfg.train_variations:
            self.env_variations = cfg.train_variations
        else:
            self.env_variations = None

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device,
                                          self.cfg.agent_type)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

        self.train_positions_set = False
        self.test_positions_set = False

    def ball_in_cup_fix_pos(self):
        "Fixes start position for ball_in_cup task. Use only on environment reset."
        assert "ball_in_cup" in self.cfg.env

        if self.step < self.cfg.num_train_steps:
            penetrating = True
            while penetrating:
                ball_x = np.random.uniform(-0.1, 0.1)
                ball_z = np.random.uniform(0.4, 0.5)
                self.env.physics.named.data.qpos['ball_x'] = ball_x
                self.env.physics.named.data.qpos['ball_z'] = ball_z
                # Check for collisions.
                self.env.physics.after_reset()
                penetrating = self.env.physics.data.ncon > 0
        else:
            penetrating = True
            while penetrating:
                x = np.random.choice([0,1])
                if x == 0:
                    ball_x = np.random.uniform(-0.2, -0.1)
                else:
                    ball_x = np.random.uniform(0.1, 0.2)
                ball_z = np.random.uniform(0.2, 0.4)
                self.env.physics.named.data.qpos['ball_x'] = ball_x
                self.env.physics.named.data.qpos['ball_z'] = ball_z
                # Check for collisions.
                self.env.physics.after_reset()
                penetrating = self.env.physics.data.ncon > 0

        self.env.task.after_step(self.env.physics)
        observation = self.env.task.get_observation(self.env.physics)
        import dm_env
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)
        obs = self.env.unwrapped._get_obs(timestep)

        frames = [obs]*self.cfg.frame_stack
        obs = np.concatenate(list(frames), axis=0)
        self.env.custom_reset(frames)
        return obs

    def reacher_fix_pos(self, eval=False):
        "Fixes position of target for reacher task. Use only on environment reset."
        assert "reacher" in self.cfg.env

        # Randomly select 2 sufficiently different positions for train and test
        if not self.train_positions_set:
            self.train_angles = [np.random.uniform(0, 0.5*np.pi), np.random.uniform(np.pi, 1.5*np.pi)]
            self.train_radiuses = np.random.uniform(0.05, 0.20, size=2)
            self.train_positions_set = True
        if not self.test_positions_set:
            self.test_angles = [np.random.uniform(0.5*np.pi, np.pi), np.random.uniform(1.5*np.pi, 2*np.pi)]
            self.test_radiuses = np.random.uniform(0.05, 0.20, size=2)
            self.test_positions_set = True

        rnd_idx = np.random.choice([0, 1])

        if self.step < self.cfg.num_train_steps:
            angle = self.train_angles[rnd_idx]
            radius = self.train_radiuses[rnd_idx]
        else:
            angle = self.test_angles[rnd_idx]
            radius = self.test_radiuses[rnd_idx]

        self.env.physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
        self.env.physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

        self.env.physics.after_reset()
        self.env.task.after_step(self.env.physics)
        observation = self.env.task.get_observation(self.env.physics)
        import dm_env
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)
        obs = self.env.unwrapped._get_obs(timestep)

        frames = [obs]*self.cfg.frame_stack
        obs = np.concatenate(list(frames), axis=0)
        self.env.custom_reset(frames)
        return obs

    def evaluate(self):
        average_episode_reward = 0

        for episode in range(self.cfg.num_eval_episodes):

            if self.env_variations:
                curr_eval_var = random.choice(self.env_variations)
                xml_path = os.path.join(get_original_cwd(), 'world_models', curr_eval_var)
                self.env.physics.reload_from_xml_path(xml_path)

            obs = self.env.reset()

            if self.cfg.reacher_fix_pos:
                obs = self.reacher_fix_pos()
            if self.cfg.ball_in_cup_fix_pos:
                obs = self.ball_in_cup_fix_pos()

            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                obs, reward, done, info = self.env.step(action)

                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        total_num_steps = self.cfg.num_train_steps + self.cfg.num_test_steps

        while self.step <= total_num_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                if self.env_variations:
                    curr_eval_var = random.choice(self.env_variations)
                    xml_path = os.path.join(get_original_cwd(), 'world_models', curr_eval_var)
                    self.env.physics.reload_from_xml_path(xml_path)

                obs = self.env.reset()

                if self.cfg.reacher_fix_pos:
                    obs = self.reacher_fix_pos()
                if self.cfg.ball_in_cup_fix_pos:
                    obs = self.ball_in_cup_fix_pos()

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            if self.step % self.cfg.save_frequency == 0:
                saveables = {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "critic_target": self.agent.critic_target.state_dict()
                }
                if self.cfg.ted:
                    saveables["ted_classifier"] = self.agent.ted_classifier.state_dict()
                save_at = os.path.join(self.save_dir, f"env_step{self.step * self.cfg.action_repeat}")
                os.makedirs(save_at, exist_ok=True)
                torch.save(saveables, os.path.join(save_at, "models.pt"))

                artifact = wandb.Artifact(f'{self.cfg.agent_type}{"_ted" if self.cfg.ted else ""}_{wandb.run.id}', type="model")
                artifact.add_file(os.path.join(save_at, "models.pt"), "models.pt")
                wandb.log_artifact(artifact)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max, episode)

            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step == self.cfg.num_train_steps:
                self.env = self.test_env

                if self.cfg.test_variations:
                    self.env_variations = self.cfg.test_variations
                else:
                    self.env_variations = None

                done = True


@hydra.main(config_path='config/config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    global workspace
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
