import sys, os
from diffusion import Diffusion
from diffusion import DiffusionSAC
from diffusion.model import MLP, DoubleCritic
from tianshou.data import Batch
from tianshou.data import VectorReplayBuffer

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)

parent_path_1 = os.path.dirname(parent_path)
sys.path.append(parent_path_1)

import torch
import datetime
import argparse

from env import environment
from env.config import VehicularEnvConfig

from env.utils import plot_rewards, save_args, plot_completion_rate
from env.utils import save_results_1, make_dir


def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DiffRL', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Multihop-V2V', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=1000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")

    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--hidden-sizes', type=int, default=256)
    parser.add_argument('--max-action', type=int, default=10)
    parser.add_argument('-t', '--n-timesteps', type=int, default=20)
    parser.add_argument('--beta-schedule', type=str, default='vp', choices=['linear', 'cosine', 'vp'])
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--pg-coef', type=float, default=1.)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument("--DiffRL-start-learn",
                        type=int,
                        default=512,
                        help="Iteration start Learn for DiffRL")
    parser.add_argument("--DiffRL-learn-interval",
                        type=int,
                        default=2,
                        help="DiffRL's learning interval")

    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    return args


def env_agent_config(cfg, seed=1):
    env = environment.RoadState()
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"n states: {n_states}, n actions: {n_actions}")

    actor_net = MLP(
        state_dim=n_states,
        action_dim=n_actions,
        hidden_dim=cfg.hidden_sizes
    )
    actor = Diffusion(
        state_dim=n_states,
        action_dim=n_actions,
        model=actor_net,
        max_action=cfg.max_action,
        beta_schedule=cfg.beta_schedule,
        n_timesteps=cfg.n_timesteps
    ).to(cfg.device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=cfg.actor_lr,
        weight_decay=cfg.wd
    )

    critic = DoubleCritic(
        state_dim=n_states,
        action_dim=n_actions,
        hidden_dim=cfg.hidden_sizes
    ).to(cfg.device)
    critic_optim = torch.optim.Adam(
        critic.parameters(),
        lr=cfg.critic_lr,
        weight_decay=cfg.wd
    )
    agent = DiffusionSAC(
        actor,
        actor_optim,
        n_states,
        critic,
        critic_optim,
        torch.distributions.Categorical,
        cfg.device,
        alpha=cfg.alpha,
        tau=cfg.tau,
        gamma=cfg.gamma,
        estimation_step=cfg.n_step,
        lr_decay=cfg.lr_decay,
        pg_coef=cfg.pg_coef,
        action_space=n_actions
    )

    return env, agent


def train(cfg, env, agent, buffer):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algo：{cfg.algo_name}, Device：{cfg.device}')
    rewards_plot = []
    ma_rewards_plot = []
    offloading_vehicle_number_plot = []
    offloading_rsu_number_plot = []
    offloading_cloud_number_plot = []
    completion_rate_plot = []
    ma_completion_rate_plot = []
    count = 0
    for i_ep in range(cfg.train_eps):
        rewards = 0
        steps = 0
        offloading_vehicle_number = 0
        offloading_rsu_number = 0
        offloading_cloud_number = 0
        complete_number = 0
        state, function = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        while True:
            steps += 1
            count += 1
            out = agent.forward(Batch(obs=state_tensor), state=None)
            action = out.act[0].item()

            next_state, reward, done, next_function, offloading_vehicle, offloading_rsu, offloading_cloud, complete = env.step(
                action, function)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            buffer.add(Batch(
                obs=[state_tensor],
                act=[action],
                rew=[reward],
                obs_next=[next_state_tensor],
                terminated=[done],
                truncated=[complete],
            ))
            state_tensor = next_state_tensor
            function = next_function
            rewards += reward
            offloading_vehicle_number += offloading_vehicle
            offloading_rsu_number += offloading_rsu
            offloading_cloud_number += offloading_cloud
            complete_number += complete
            if (count > cfg.DiffRL_start_learn) and (count % cfg.DiffRL_learn_interval == 0):
                agent.update(sample_size=cfg.batch_size, buffer=buffer)

            if done:
                break
        rewards_plot.append(rewards)
        offloading_vehicle_number_plot.append(offloading_vehicle_number)
        offloading_rsu_number_plot.append(offloading_rsu_number)
        offloading_cloud_number_plot.append(offloading_cloud_number)
        completion_rate = complete_number / (
                VehicularEnvConfig().rsu_number * (VehicularEnvConfig().time_slot_end + 1))
        completion_rate_plot.append(completion_rate)
        print("#  episode :{}, steps : {}, rewards : {}, complete : {}, vehicle : {}, rsu : {}, cloud : {}"
              .format(i_ep + 1, steps, rewards,
                      completion_rate, offloading_vehicle_number, offloading_rsu_number, offloading_cloud_number))
        if ma_rewards_plot:
            ma_rewards_plot.append(0.9 * ma_rewards_plot[-1] + 0.1 * rewards)
        else:
            ma_rewards_plot.append(rewards)

        if ma_completion_rate_plot:
            ma_completion_rate_plot.append(0.9 * ma_completion_rate_plot[-1] + 0.1 * completion_rate)
        else:
            ma_completion_rate_plot.append(completion_rate)

    res_dic_rewards = {'rewards': rewards_plot, 'ma_rewards': ma_rewards_plot}
    res_dic_completion_rate = {'completion_rate': completion_rate_plot,
                               'ma_completion_rate': ma_completion_rate_plot}
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    save_results_1(res_dic_rewards, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_completion_rate, tag='train',
                   path=cfg.result_path)
    plot_rewards(res_dic_rewards['rewards'], res_dic_rewards['ma_rewards'], cfg, tag="train")
    plot_completion_rate(res_dic_completion_rate['completion_rate'], res_dic_completion_rate['ma_completion_rate'],
                         cfg, tag="train")
    env.close()


if __name__ == "__main__":
    cfg = get_args()
    buffer = VectorReplayBuffer(
        total_size=cfg.buffer_size,
        buffer_num=1
    )
    env, agent = env_agent_config(cfg)
    train(cfg, env, agent, buffer)
    save_args(cfg)
