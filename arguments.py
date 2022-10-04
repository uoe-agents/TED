import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--episode_length', default=1000, type=int)

    # train
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--algorithm', default='rad', type=str)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--num_train_steps', default=200000, type=int)
    parser.add_argument('--num_test_steps', default=200000, type=int)
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--num_train_iters', default=1, type=int)
    parser.add_argument('--num_seed_steps', default=1000, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--seed', default=0, type=int)

    # observation
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)

    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)

    # log
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--save_freq', default=50000, type=int)
    parser.add_argument('--log_dir', default='runs', type=str)
    parser.add_argument('--save_video', default="False", type=str)

    # train distractors
    parser.add_argument('--difficulty', default=None)
    parser.add_argument('--dynamic', default=False)
    parser.add_argument('--camera_kwargs', default=None)
    parser.add_argument('--colour_kwargs', default="{'max_delta': 0.1,'step_std': 0.0, 'disjoint_sets': 'train'}", type=str)
    parser.add_argument('--background_kwargs', default=None)
    parser.add_argument('--background_dataset_path', default="None", type=str)
    parser.add_argument('--background_dataset_videos', default="train")

    # test distractors
    parser.add_argument('--test_difficulty', default=None)
    parser.add_argument('--test_dynamic', default=False)
    parser.add_argument('--test_camera_kwargs', default=None)
    parser.add_argument('--test_colour_kwargs', default="{'max_delta': 0.1,'step_std': 0.0, 'disjoint_sets': 'test'}", type=str)
    parser.add_argument('--test_background_kwargs', default=None)
    parser.add_argument('--test_background_dataset_path', default="None", type=str)
    parser.add_argument('--test_background_dataset_videos', default="val")

    # agent
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--hidden_depth', default=2, type=int)

    # TED
    parser.add_argument('--ted', default="True", type=str)
    parser.add_argument('--ted_coef', default=1)
    parser.add_argument('--ted_lr', default=1e-3, type=float)

    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)

    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # encoder
    parser.add_argument('--num_conv_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--feature_dim', default=50, type=int)
    parser.add_argument('--encoder_tau', default=0.01, type=float)

    # svea
    parser.add_argument('--svea_alpha', default=0.5, type=float)
    parser.add_argument('--svea_beta', default=0.5, type=float)

    args = parser.parse_args()

    args.ted = eval(args.ted)
    args.save_video = eval(args.save_video)
    args.colour_kwargs = eval(args.colour_kwargs)
    args.test_colour_kwargs = eval(args.test_colour_kwargs)

    return args