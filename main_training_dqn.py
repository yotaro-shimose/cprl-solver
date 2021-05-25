
from src.real_world.trainer_dqn import RealTrainerDQN
from src.real_world.conversion import to_virtual_time
from src.real_world.conversion import EIGHT_HOURS
from src.real_world.property import RealDQNArgs
import sys
import os
import argparse

sys.path.append(os.path.join(sys.path[0], '..'))


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_city', type=int, default=20)
    parser.add_argument('--grid_size', type=int, default=100)
    parser.add_argument('--tw_size', type=int,
                        default=to_virtual_time(EIGHT_HOURS / 4))
    parser.add_argument('--max_travel_time', type=int,
                        default=to_virtual_time(EIGHT_HOURS))
    parser.add_argument('--tw_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=-1)

    # Hyper parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_step', type=int, default=-1)
    parser.add_argument('--max_softmax_beta', type=int,
                        default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='dimension of latent layers')

    # Argument for Trainer
    parser.add_argument('--n_episode', type=int, default=1000000)
    parser.add_argument('--save_dir', type=str,
                        default='./real-result-default')
    parser.add_argument('--plot_training', type=bool, default=1)
    parser.add_argument('--mode', default='gpu', help='cpu/gpu')

    args = parser.parse_args()

    return RealDQNArgs(
        args.n_city,
        args.grid_size,
        args.tw_size,
        args.max_travel_time,
        args.tw_ratio,
        args.seed,
        args.batch_size,
        args.learning_rate,
        args.hidden_layer,
        args.latent_dim,
        args.max_softmax_beta,
        args.n_step,
        args.n_episode,
        args.mode,
        args.plot_training,
        args.save_dir,
    )


if __name__ == '__main__':

    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] TRAINING ON RANDOM INSTANCES: TSPTW")
    print("[INFO] n_city: %d" % args.n_city)
    print("[INFO] grid_size: %d" % args.grid_size)
    print("[INFO] tw_size: %d" % args.tw_size)
    print("[INFO] max_travel_time: %d" % args.max_travel_time)
    print("[INFO] tw_ratio: %f" % args.tw_ratio)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")
    print("[INFO] TRAINING PARAMETERS")
    print("[INFO] algorithm: DQN")
    print("[INFO] batch_size: %d" % args.batch_size)
    print("[INFO] learning_rate: %f" % args.learning_rate)
    print("[INFO] hidden_layer: %d" % args.hidden_layer)
    print("[INFO] latent_dim: %d" % args.latent_dim)
    print("[INFO] softmax_beta: %d" % args.max_softmax_beta)
    print("[INFO] n_step: %d" % args.n_step)
    print("[INFO] n_episode: %d" % args.n_episode)
    print("[INFO] mode: %s" % args.mode)
    print("[INFO] plot_training: %s" % args.plot_training)
    print("[INFO] plot_training: %s" % args.save_dir)
    print("***********************************************************")
    sys.stdout.flush()

    trainer = RealTrainerDQN(args)
    trainer.run_training()
