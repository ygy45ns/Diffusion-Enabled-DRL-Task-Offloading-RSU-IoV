import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)

sys.path.append(parent_path)

import numpy as np
import torch
import argparse
from methods.A3C.a3c import ActorCritic

from env.utils import plot_rewards, save_args, plot_completion_rate
from env.utils import save_results_1, make_dir

from env import environment
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import datetime
from env.config import VehicularEnvConfig


def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='A3C', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Multihop-V2V', type=str, help="name of environment")
    parser.add_argument('--n_train_processes', default=6, type=int, help="numbers of environments")
    parser.add_argument('--max_train_ep', default=800, type=int, help="episodes of training")
    parser.add_argument('--max_test_ep', default=1000, type=int, help="episodes of testing")
    parser.add_argument('--update_interval', default=5, type=int, help="展开轨迹数")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--learning_rate', default=0.0002, type=float, help="learning rate")
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str,
                        help="cpu or cuda")
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
    agent = ActorCritic(n_states, n_actions, cfg.hidden_dim)
    return env, agent


def train(cfg, env, agent, global_model, rank):
    agent.load_state_dict(global_model.state_dict())
    optimizer = optim.Adam(global_model.parameters(), lr=cfg.learning_rate)

    for n_epi in range(cfg.max_train_ep):
        done = False
        state, function = env.reset()
        while not done:
            state_lst, action_lst, reward_lst = [], [], []
            for t in range(cfg.update_interval):
                prob = agent.actor(torch.from_numpy(state).float())
                dist = Categorical(prob)
                action = dist.sample().item()
                next_state, reward, done, next_function, _, _, _, _ = env.step(action, function)

                state_lst.append(state)
                action_lst.append([action])
                reward_lst.append(reward)
                state = next_state
                function = next_function
                if done:
                    break

            final_state = torch.tensor(next_state, dtype=torch.float)
            V = 0.0 if done else agent.critic(final_state).item()
            td_target_lst = []
            for reward in reward_lst[::-1]:
                V = cfg.gamma * V + reward
                td_target_lst.append([V])
            td_target_lst.reverse()

            state_lst = np.array(state_lst, dtype=np.float32)
            action_lst = np.array(action_lst, dtype=np.int64)
            td_target_lst = np.array(td_target_lst, dtype=np.float32)

            state_batch, action_batch, td_target = torch.tensor(state_lst, dtype=torch.float), \
                torch.tensor(action_lst), torch.tensor(td_target_lst)
            advantage = td_target - agent.critic(state_batch)

            action_prob = agent.actor(state_batch, softmax_dim=1)
            all_action_prob = action_prob.gather(1, action_batch)
            loss = -torch.log(all_action_prob) * advantage.detach() + \
                   F.smooth_l1_loss(agent.critic(state_batch), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), agent.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            agent.load_state_dict(global_model.state_dict())
    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def test_global_model(cfg, env, global_model):
    rewards_plot = []
    ma_rewards_plot = []
    offloading_vehicle_number_plot = []
    offloading_rsu_number_plot = []
    offloading_cloud_number_plot = []
    completion_rate_plot = []
    ma_completion_rate_plot = []
    for n_epi in range(cfg.max_test_ep):
        rewards = 0
        steps = 0
        done = False
        offloading_vehicle_number = 0
        offloading_rsu_number = 0
        offloading_cloud_number = 0
        complete_number = 0
        state, function = env.reset()
        while not done:
            prob = global_model.actor(torch.from_numpy(state).float())
            action = Categorical(prob).sample().item()
            next_state, reward, done, next_function, offloading_vehicle, offloading_rsu, offloading_cloud, complete = env.step(
                action, function)
            state = next_state
            function = next_function
            steps += 1
            rewards += reward
            offloading_vehicle_number += offloading_vehicle
            offloading_rsu_number += offloading_rsu
            offloading_cloud_number += offloading_cloud
            complete_number += complete
        rewards_plot.append(rewards)
        offloading_vehicle_number_plot.append(offloading_vehicle_number)
        offloading_rsu_number_plot.append(offloading_rsu_number)
        offloading_cloud_number_plot.append(offloading_cloud_number)
        completion_rate = complete_number / (VehicularEnvConfig().rsu_number * (VehicularEnvConfig().time_slot_end + 1))
        completion_rate_plot.append(completion_rate)

        print("#  episode :{}, steps : {}, rewards : {}, complete : {}, vehicle : {}, rsu : {}, cloud : {}"
              .format(n_epi + 1, steps, rewards,
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
    res_dic_completion_rate = {'completion_rate': completion_rate_plot, 'ma_completion_rate': ma_completion_rate_plot}
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    save_results_1(res_dic_rewards, tag='train',
                   path=cfg.result_path)
    save_results_1(res_dic_completion_rate, tag='train',
                   path=cfg.result_path)
    plot_rewards(res_dic_rewards['rewards'], res_dic_rewards['ma_rewards'], cfg, tag="train")
    plot_completion_rate(res_dic_completion_rate['completion_rate'], res_dic_completion_rate['ma_completion_rate'], cfg,
                         tag="train")
    env.close()


if __name__ == '__main__':
    cfg = get_args()
    make_dir(cfg.result_path, cfg.model_path)
    env, global_model = env_agent_config(cfg)
    global_model.share_memory()
    result_queue = mp.Queue()
    processes = []
    for rank in range(cfg.n_train_processes + 1):
        if rank == 0:
            _, agent = env_agent_config(cfg)
            p = mp.Process(target=test_global_model, args=(cfg, env, global_model))
        else:
            _, agent = env_agent_config(cfg)
            p = mp.Process(target=train, args=(cfg, env, agent, global_model, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    save_args(cfg)
    global_model.save(path=cfg.model_path)
