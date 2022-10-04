from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.drq import DrQ
from algorithms.svea import SVEA

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'drq': DrQ,
	'svea': SVEA
}

def make_agent(obs_shape, action_shape, action_range, cfg):
	return algorithm[cfg.algorithm](obs_shape, action_shape, action_range, cfg)