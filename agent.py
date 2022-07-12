import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import utils
import hydra


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        h = self.forward_conv(obs)

        if detach_encoder_conv:
            h = h.detach()

        out = self.head(h)

        if not self.output_logits:
            out = torch.tanh(out)

        if detach_encoder_head:
            out = out.detach()
        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def copy_head_weights_from(self, source):
        """Tie head layers"""
        for i in range(2):
            utils.tie_weights(src=source.head[i], trg=self.head[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)

class TEDClassifier(nn.Module):
    """TED classifer to predict if the input pair is temporal or non-temporal."""
    def __init__(self, hidden_size):
        super().__init__()

        self.W = nn.Parameter(torch.empty(2, hidden_size))
        self.b = nn.Parameter(torch.empty((1, hidden_size)))
        self.W_bar = nn.Parameter(torch.empty((1, hidden_size)))
        self.b_bar = nn.Parameter(torch.empty((1, hidden_size)))
        self.c = nn.Parameter(torch.empty((1, 1)))

        self.W.requires_grad = True
        self.b.requires_grad = True
        self.W_bar.requires_grad = True
        self.b_bar.requires_grad = True
        self.c.requires_grad = True

        nn.init.orthogonal_(self.W)
        nn.init.orthogonal_(self.b)
        nn.init.orthogonal_(self.W_bar)
        nn.init.orthogonal_(self.b_bar)
        nn.init.orthogonal_(self.c)

    def forward(self, inputs):

        x = self.W * inputs
        x = torch.sum(x, dim=1)
        x = x + self.b
        x = torch.abs(x)

        y = torch.square((self.W_bar * torch.transpose(inputs, 1, 0)[0]) + self.b_bar)

        output = (torch.sum((x-y), dim=1) + self.c).squeeze()

        return output

class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds

        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()

        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        obs = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employs double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth, device):
        super().__init__()
        self.device = device
        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()

        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder_conv=False, detach_encoder_head=False):

        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)

        obs_action = torch.cat([obs, action], dim=-1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class Agent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, agent_type, obs_shape, action_shape, action_range, device, encoder_cfg,
                 critic_cfg, actor_cfg, ted_cfg, discount, init_temperature, lr,
                 encoder_lr, actor_update_frequency, critic_tau, encoder_tau,
                 critic_target_update_frequency, batch_size, ted, ted_coef):

        self.agent_type = agent_type
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.ted = ted
        self.ted_coef = ted_coef
        if self.ted:
            self.ted_classifier = hydra.utils.instantiate(ted_cfg).to(self.device)
            # share head network weights
            self.actor.encoder.copy_head_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        if self.ted:
            # do not update encoder params with critic optimizer (retain gradient instead)
            critic_params = list(self.critic.Q1.parameters()) + list(self.critic.Q2.parameters())
            self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr)
            ted_params = list(self.ted_classifier.parameters()) + list(self.critic.encoder.parameters())
            self.ted_optimizer = torch.optim.Adam(ted_params, lr=encoder_lr)
            self.ted_loss = nn.BCEWithLogitsLoss()
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

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


    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            if self.agent_type == "drq":
                dist_aug = self.actor(next_obs_aug)
                next_action_aug = dist_aug.rsample()
                log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
                target_Q1, target_Q2 = self.critic_target(next_obs_aug, next_action_aug)
                target_V = torch.min(
                    target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
                target_Q_aug = reward + (not_done * self.discount * target_V)

                target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.agent_type == "drq":
            Q1_aug, Q2_aug = self.critic(obs_aug, action)

            critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

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

    def update_representation(self, obs, next_obs, idxs, replay_buffer, logger, step):
        num_samples = obs.shape[0]

        obs_rep = self.critic.encoder(obs)
        with torch.no_grad():
            next_obs_rep = self.critic_target.encoder(next_obs)

        samples = []
        labels = []
        for i in range(num_samples):
            # Stack the consecutive observations to make a temporal sample
            non_iid_sample = torch.stack([obs_rep[i], next_obs_rep[i]])
            samples.append(non_iid_sample)
            # All temporal samples are given a label of 0
            labels.append(torch.tensor(1.0))

            # Create the non-temporal different episode sample
            rnd_idx = random.choice([n for n in range(num_samples) if n != i])
            iid_sample = torch.stack([obs_rep[i], next_obs_rep[rnd_idx]])
            samples.append(iid_sample)
            # All non-temporal samples are given a label of 1
            labels.append(torch.tensor(0.0))

            # Create non-temporal same episode sample
            sample_idx = idxs[i]
            sample_episode = replay_buffer.episode[sample_idx]
            try:
                all_same_episode_idxs = np.nonzero(replay_buffer.episode == sample_episode)[0]
                idx_to_remove = 1 - np.isin(all_same_episode_idxs, [sample_idx - 1, sample_idx, sample_idx + 1])
                x = all_same_episode_idxs * idx_to_remove
                reduced_same_episode_idxs = list(x[x != 0])
                idx_from_same_episode = random.choice(reduced_same_episode_idxs)
            except:
                # skip samples for which there are no other samples from same episode in replay buffer
                idx_from_same_episode = None

            if idx_from_same_episode:
                next_entry = torch.as_tensor(replay_buffer.next_obses[idx_from_same_episode],
                                             device=self.device).float()
                next_entry = next_entry.unsqueeze(0)
                with torch.no_grad():
                    next_entry_rep = self.critic_target.encoder(next_entry)
                iid_sample_same_episode = torch.stack([obs_rep[i], next_entry_rep.squeeze()])
                samples.append(iid_sample_same_episode)
                labels.append(torch.tensor(0.0))

        samples = torch.stack(samples)
        labels = torch.stack(labels).to(self.device)

        r = self.ted_classifier(samples)
        ted_loss = self.ted_loss(r, labels) * self.ted_coef

        logger.log('train_ted/loss', ted_loss, step)

        ted_loss.backward()
        self.ted_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug, indexes = replay_buffer.sample(
            self.batch_size)

        if self.ted:
            # Zero grad here as we will retain critic gradient
            self.ted_optimizer.zero_grad()

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if self.ted:
            self.update_representation(obs, next_obs, indexes, replay_buffer, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)