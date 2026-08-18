import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)

parent_path_1 = os.path.dirname(parent_path)
sys.path.append(parent_path_1)

import torch
import datetime
import argparse

from methods.DQN.dqn import DQN
from env import environment
from env.config import VehicularEnvConfig

from env.utils import plot_rewards, save_args, plot_completion_rate
from env.utils import save_results_1, make_dir


def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DQN', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Multihop-V2V', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=1000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.9, type=float, help="discounted factor")
    parser.add_argument('--epsilon_start', default=1.0, type=float, help="initial value of epsilon")
    parser.add_argument('--epsilon_end', default=0.01, type=float, help="final value of epsilon")
    parser.add_argument('--epsilon_decay', default=6000, type=int, help="decay rate of epsilon")
    parser.add_argument('--lr', default=0.0005, type=float, help="learning rate")
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--target_update', default=4, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
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
    agent = DQN(n_states, n_actions, cfg)
    return env, agent


def train(cfg, env, agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algo：{cfg.algo_name}, Device：{cfg.device}')
    rewards_plot = []
    ma_rewards_plot = []
    offloading_vehicle_number_plot = []
    offloading_rsu_number_plot = []
    offloading_cloud_number_plot = []
    completion_rate_plot = []
    ma_completion_rate_plot = []

    for i_ep in range(cfg.train_eps):
        rewards = 0
        steps = 0
        offloading_vehicle_number = 0
        offloading_rsu_number = 0
        offloading_cloud_number = 0
        complete_number = 0
        state, function = env.reset()
        while True:
            steps += 1
            action = agent.choose_action(state)
            next_state, reward, done, next_function, offloading_vehicle, offloading_rsu, offloading_cloud, complete = env.step(
                action, function)
            agent.memory.push(state, action, reward,
                              next_state, done)
            state = next_state
            function = next_function
            agent.update()
            rewards += reward
            offloading_vehicle_number += offloading_vehicle
            offloading_rsu_number += offloading_rsu
            offloading_cloud_number += offloading_cloud
            complete_number += complete
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

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
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    agent.save(cfg.model_path)
    env.close()


if __name__ == "__main__":
    cfg = get_args()
    env, agent = env_agent_config(cfg)
    train(cfg, env, agent)
