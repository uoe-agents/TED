import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from algorithms import modules

class SAC(object):
    def __init__(self, obs_shape, action_shape, action_range, cfg):

        self.cfg = cfg
        self.action_range = action_range
        self.device = cfg.device
        self.discount = cfg.discount
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.actor_update_frequency = cfg.actor_update_freq
        self.critic_target_update_frequency = cfg.critic_target_update_freq
        self.batch_size = cfg.batch_size

        self.actor = modules.Actor(obs_shape, action_shape, cfg).to(self.device)

        self.critic = modules.Critic(obs_shape, action_shape, cfg).to(self.device)
        self.critic_target = modules.Critic(obs_shape, action_shape, cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.ted = cfg.ted
        self.ted_coef = cfg.ted_coef
        if self.ted:
            self.ted_classifier = modules.TEDClassifier(cfg).to(self.device)
            # share head network weights
            self.actor.encoder.copy_head_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        if self.ted:
            # do not update encoder params with critic optimizer (retain gradient instead)
            critic_params = list(self.critic.Q1.parameters()) + list(self.critic.Q2.parameters())
            self.critic_optimizer = torch.optim.Adam(critic_params, lr=cfg.critic_lr)
            ted_params = list(self.ted_classifier.parameters()) + list(self.critic.encoder.parameters())
            self.ted_optimizer = torch.optim.Adam(ted_params, lr=cfg.ted_lr)
            self.ted_loss = nn.BCEWithLogitsLoss()
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.ted:
            self.ted_classifier.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)

        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])


    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=self.ted)
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder_conv=True, detach_encoder_head=self.ted)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder_conv=True, detach_encoder_head=self.ted)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_representation(self, obs, next_obs, same_episode_obs, replay_buffer, logger, step):
        num_samples = obs.shape[0]

        obs_rep = self.critic.encoder(obs)
        with torch.no_grad():
            next_obs_rep = self.critic_target.encoder(next_obs)

        # Stack the consecutive observations to make temporal samples
        non_iid_samples = torch.stack([obs_rep, next_obs_rep], dim=1)
        # All temporal samples are given a label of 1
        non_iid_labels = torch.ones((num_samples))

        # Create the non-temporal different episode samples
        rnd_idx = torch.randperm(num_samples)
        diff_ep_iid_samples = torch.stack([obs_rep, next_obs_rep[rnd_idx]], dim=1)
        # All non-temporal samples are given a label of 0
        diff_ep_iid_labels = torch.zeros((num_samples))

        # Create the non-temporal same episode samples
        with torch.no_grad():
            next_entry_rep = self.critic_target.encoder(same_episode_obs)
        same_ep_iid_samples = torch.stack([obs_rep, next_entry_rep], dim=1)
        same_ep_iid_labels = torch.zeros((num_samples))

        samples = torch.cat([non_iid_samples, diff_ep_iid_samples, same_ep_iid_samples])
        labels = torch.cat([non_iid_labels, diff_ep_iid_labels, same_ep_iid_labels]).to(self.device)

        r = self.ted_classifier(samples)
        ted_loss = self.ted_loss(r, labels) * self.ted_coef

        logger.log('train_ted/loss', ted_loss, step)

        ted_loss.backward()
        self.ted_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, same_episode_obs = replay_buffer.sample(self.batch_size)

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