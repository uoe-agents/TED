import torch
import torch.nn as nn
import utils

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, cfg):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = cfg.num_conv_layers
        self.num_filters = cfg.num_filters
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = cfg.feature_dim

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2)])
        for i in range(1, self.num_layers):
            self.convs.extend([nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)])

        # get output shape
        x = torch.randn(*obs_shape).unsqueeze(0)
        conv = torch.relu(self.convs[0](x))
        for i in range(1, self.num_layers):
            conv = self.convs[i](conv)
        conv = conv.view(conv.size(0), -1)
        self.output_shape = conv.shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.output_shape, self.feature_dim),
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
    def __init__(self, cfg):
        super().__init__()

        self.W = nn.Parameter(torch.empty(2, cfg.feature_dim))
        self.b = nn.Parameter(torch.empty((1, cfg.feature_dim)))
        self.W_bar = nn.Parameter(torch.empty((1, cfg.feature_dim)))
        self.b_bar = nn.Parameter(torch.empty((1, cfg.feature_dim)))
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
    def __init__(self, obs_shape, action_shape, cfg):
        super().__init__()

        self.encoder = Encoder(obs_shape, cfg)

        self.log_std_bounds = [cfg.actor_log_std_min, cfg.actor_log_std_max]

        self.trunk = utils.mlp(self.encoder.feature_dim, cfg.hidden_dim,
                               2 * action_shape[0], cfg.hidden_depth)

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
    def __init__(self, obs_shape, action_shape, cfg):
        super().__init__()
        self.encoder = Encoder(obs_shape, cfg)

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            cfg.hidden_dim, 1, cfg.hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            cfg.hidden_dim, 1, cfg.hidden_depth)

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