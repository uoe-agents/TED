import os
import time
import dmc2gym
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
import algorithms
from arguments import parse_args
import datetime

torch.backends.cudnn.benchmark = True

def make_env(cfg, test=False):
    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if cfg.domain_name == 'quadruped' else 0

    if test:
        env = dmc2gym.make(domain_name=cfg.domain_name,
                           task_name=cfg.task_name,
                           test=test,
                           seed=cfg.seed,
                           difficulty=None if cfg.test_difficulty == "None" else cfg.test_difficulty,
                           dynamic=cfg.test_dynamic,
                           background_dataset_path=None if cfg.test_background_dataset_path == "None" else os.path.join(os.getcwd(), cfg.test_background_dataset_path),
                           background_dataset_videos=None if cfg.test_background_dataset_videos == "None" else cfg.test_background_dataset_videos,
                           background_kwargs=None if cfg.test_background_kwargs == "None" else cfg.test_background_kwargs,
                           camera_kwargs=None if cfg.test_camera_kwargs == "None" else cfg.test_camera_kwargs,
                           color_kwargs=None if cfg.test_colour_kwargs == "None" else cfg.test_colour_kwargs,
                           visualize_reward=False,
                           from_pixels=True,
                           height=cfg.image_size,
                           width=cfg.image_size,
                           frame_skip=cfg.action_repeat,
                           camera_id=camera_id,
                           channels_first=True)

    else:
        env = dmc2gym.make(domain_name=cfg.domain_name,
                           task_name=cfg.task_name,
                           test=test,
                           seed=cfg.seed,
                           difficulty=None if cfg.difficulty == "None" else cfg.difficulty,
                           dynamic=cfg.dynamic,
                           background_dataset_path=None if cfg.background_dataset_path == "None" else os.path.join(os.getcwd(), cfg.background_dataset_path),
                           background_dataset_videos=None if cfg.background_dataset_videos == "None" else cfg.background_dataset_videos,
                           background_kwargs=None if cfg.background_kwargs == "None" else cfg.background_kwargs,
                           camera_kwargs=None if cfg.camera_kwargs == "None" else cfg.camera_kwargs,
                           color_kwargs=None if cfg.colour_kwargs == "None" else cfg.colour_kwargs,
                           visualize_reward=False,
                           from_pixels=True,
                           height=cfg.image_size,
                           width=cfg.image_size,
                           frame_skip=cfg.action_repeat,
                           camera_id=camera_id,
                           channels_first=True)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        algo = f"{cfg.algorithm}_ted_coef{cfg.ted_coef}" if cfg.ted else cfg.algorithm
        exp_folder = cfg.exp_name if cfg.exp_name else datetime.date.today().strftime(("%d-%b-%Y"))
        self.work_dir = os.path.join(os.getcwd(), cfg.log_dir, exp_folder, algo, str(cfg.seed))
        assert not os.path.exists(self.work_dir), 'specified working directory already exists'
        os.makedirs(self.work_dir)
        print(f'workspace: {self.work_dir}')
        self.save_dir = os.path.join(self.work_dir, "trained_models")
        utils.write_info(cfg, os.path.join(self.work_dir, 'config.log'))

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=False,
                             log_frequency=cfg.log_freq,
                             action_repeat=cfg.action_repeat,
                             agent=cfg.algorithm)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.train_env = make_env(cfg, test=False)
        self.env = self.train_env

        self.test_env = make_env(cfg, test=True)

        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = algorithms.make_agent(self.env.observation_space.shape, self.env.action_space.shape, action_range, cfg)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device,
                                          self.cfg.ted)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0

        for episode in range(self.cfg.num_eval_episodes):

            obs = self.env.reset()

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
                if self.step % self.cfg.eval_freq == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                obs = self.env.reset()

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

            if self.step > 0 and self.step % self.cfg.save_freq == 0:
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
                if self.cfg.collect_disentanglement_samples:
                    print("Collecting disentanglement metric data")
                    import collect_disentanglement_metric_data
                    collect_disentanglement_metric_data.main(self.cfg, self.agent, self.device)

                print("Switching to test env")
                self.env = self.test_env

                done = True

def main(cfg):
    from train import Workspace as W
    global workspace
    workspace = W(cfg)

    start_time = time.time()
    workspace.run()
    print("total run time: ", time.time()-start_time)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
