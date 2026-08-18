import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties


def chinese_font():
    try:
        font = FontProperties(
            fname='C:/Windows/Fonts/STSONG.TTF', size=15)
    except:
        font = None
    return font


def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()
    plt.title("learning curve on {} of {}".format(cfg.device, cfg.algo_name), fontsize=18)
    plt.xlabel('epsiodes', fontsize=18)
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    plt.grid()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()


def plot_completion_rate(completion_rate, ma_completion_rate, cfg, tag='train'):
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.figure()
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')
    plt.xlabel('episodes', fontsize=18, fontname='Times New Roman')
    plt.ylabel('completion ratio', fontsize=18, fontname='Times New Roman')
    plt.plot(completion_rate, label='completion_rate')
    plt.plot(ma_completion_rate, label='ma_completion_rate')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(prop={'size': 18, 'family': 'Times New Roman'})
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_completion_rate_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()


def save_results_1(dic, tag='train', path='./results'):
    for key, value in dic.items():
        np.save(path + '{}_{}.npy'.format(tag, key), value)
    print('Results saved！')


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')


def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def save_args(args):
    argsDict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")