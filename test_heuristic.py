from common_utils import *
from params import configs
from Mic_env import MICEnvForVariousOpNums
from tqdm import tqdm
from data_utils import pack_data_from_config
import time
import numpy as np
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


def test_heuristic_method(data_set, heuristic, seed):
    """
        test one heuristic method on the given data
    :param data_set:  test data
    :param heuristic: the name of heuristic method
    :param seed: seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    result = []

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        n_j = data_set[0][i].shape[0]
        n_op, n_m = data_set[1][i].shape
        env = MICEnvForVariousOpNums(n_j=n_j, n_m=n_m)

        env.set_initial_data([data_set[0][i]], [data_set[1][i]])

        t1 = time.time()
        while True:
            action1, action2 = heuristic_select_action(heuristic, env)

            _, _, done, _ = env.step(np.array([action1], np.array([action2])))

            if done:
                break

        t2 = time.time()
        # tqdm.write(f'Instance {i + 1} , makespan:{-ep_reward} , time:{t2 - t1}')
        result.append([env.current_makespan[0], t2 - t1])

    return np.array(result)


def main():

    setup_seed(configs.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    test_data = pack_data_from_config(configs.data_source, configs.test_data)
    if len(configs.test_method) == 0:
        test_method = ['FIFO+SPT', 'MOPNR+SPT', 'MWKR+SPT', 'FIFO+EET', 'MOPNR+EET', 'MWKR+EET']
    else:
        test_method = configs.test_method

    for data in test_data:
        print("-" * 25 + "Test Heuristic Methods" + "-" * 25)
        print('Test Methods:', test_method)
        print(f"test data name: {configs.data_source},{data[1]}")
        save_direc = f'./test_results2/{configs.data_source}/{data[1]}'

        if not os.path.exists(save_direc):
            os.makedirs(save_direc)
        for method in test_method:
            save_path = save_direc + f'/Result_{method}_{data[1]}.npy'

            if (not os.path.exists(save_path)) or configs.cover_heu_flag:
                print(f"Heuristic method : {method}")
                seed = configs.seed_test

                num_runs = 1   # 如果以后想 5 个 seed，直接改这里
                all_results = []

                for j in range(num_runs):
                    result = test_heuristic_method(data[0], method, seed + j)
                    all_results.append(result)  # (num_instances, 2)

                all_results = np.array(all_results)  # (num_runs, num_instances, 2)

                # ---------- 保存完整 instance-level 结果 ----------
                save_path = save_direc + f'/Result_{method}_{data[1]}_all.npy'
                np.save(save_path, all_results)

                # ---------- 仍然保留你现在的“平均结果” ----------
                mean_result = np.mean(all_results, axis=0)  # (num_instances, 2)
                mean_save_path = save_direc + f'/Result_{method}_{data[1]}_mean.npy'
                np.save(mean_save_path, mean_result)

                print(f"testing results of {method}:")
                print(f"makespan (instance mean): ", mean_result[:, 0].mean())
                print(f"time: ", mean_result[:, 1].mean())



if __name__ == '__main__':
    main()
