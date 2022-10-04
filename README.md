# Temporal Disentanglement

Official implementation of Temporal Disentanglement of 
Representations for Improved Generalisation in Reinforcement Learning.

This code is based on the DrQ PyTorch implementation by [Yarats et al.](https://github.com/denisyarats/drq). As per the original code base, we use [kornia](https://github.com/kornia/kornia) for data augmentation.

Our SVEA implementation is based on the official implementation from the [DMControl Generalization Benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark).

The 'distracting_control' folder contains the [Distracting Control Suite](https://github.com/sahandrez/distracting_control) 
code with minor amendments to create disjoint colour sets for training and testing. The 'dmc2gym' folder contains the
[dmc2gym](https://github.com/denisyarats/dmc2gym) code amended to use the distracting_control wrappers.

## Requirements
We assume you have access to [MuJoCo](https://github.com/openai/mujoco-py) and a gpu that can run CUDA 10.2. Then, the simplest way to install all required 
dependencies is to create an anaconda environment by running:
```(python)
conda env create -f conda_env.yml
```
You can activate your environment with:
```(python)
conda activate ted
```

## Instructions
You can run the code using the configuration specified in config.yaml with:
```(python)
python train.py
```

You can also override the default configuration values, for example:
```(python)
python train.py --domain_name walker --task_name walk --ted True
```

This will produce the `runs` folder, where all the outputs are going to be stored including train/eval logs, 
tensorboard blobs, and evaluation episode videos. To launch tensorboard run
```
tensorboard --logdir runs
```

The console output is also available in a form:
```
| train | E: 5 | S: 5000 | R: 11.4359 | D: 66.8 s | BR: 0.0581 | ALOSS: -1.0640 | CLOSS: 0.0996 | TLOSS: -23.1683 | TVAL: 0.0945 | AENT: 3.8132 | TEDLOSS: 2.5631
```
a training entry decodes as
```
train - training episode
E - total number of episodes
S - total number of environment steps
R - episode return
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy
TEDLOSS - average of the TED auxiliary loss
```
while an evaluation entry
```
| eval  | E: 20 | S: 20000 | R: 10.9356
```
contains
```
E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)
```
