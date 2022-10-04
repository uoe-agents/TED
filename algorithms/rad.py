import utils
from algorithms.sac import SAC

class RAD(SAC):
    def __init__(self, obs_shape, action_shape, action_range, cfg):
        super().__init__(obs_shape, action_shape, action_range, cfg)

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, same_episode_obs = replay_buffer.sample_rad(self.batch_size)

        if self.ted:
            # Zero grad here as we will retain critic gradient
            self.ted_optimizer.zero_grad()

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if self.ted:
            self.update_representation(obs, next_obs, same_episode_obs, replay_buffer, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)