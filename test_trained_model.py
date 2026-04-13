import time
import os
from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import pack_data_from_config
from model.PPO import PPO_initialize
from common_utils import setup_seed
from Mic_env import MICEnvForVariousOpNums

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

ppo = PPO_initialize()
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def test_greedy_strategy(data_set, model_path, seed):
    """
        test the model on the given data using the greedy strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """

    test_result_list = []
    done_flag = 0

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.to(device)
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    env = MICEnvForVariousOpNums(n_j=n_j, n_m=n_m)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        fatigue_all = []

        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        t1 = time.time()
        while True:

            with torch.no_grad():
                pi, _ = ppo.policy(fea_mou=state.fea_mou_tensor.to(device),  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor.to(device),  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor.to(device),  # [sz_b, J]
                                                             fea_wor=state.fea_wor_tensor.to(device),  # [sz_b, M, 6]
                                                             worker_mask=state.worker_mask_tensor.to(device),  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor.to(device),  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor.to(device),
                                                             # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor.to(device),
                                                             worker_fatigue_time_tensor=state.worker_fatigue_time_tensor.to(device),
                                                             worker_memory_time_tensor=state.worker_memory_time_tensor.to(device),
                                                             done_flag=done_flag)

            action, _ = sample_action(pi)

            state, reward, done, sum_fatigue = env.step(actions=action.cpu().numpy())
            fatigue_all.append(sum_fatigue)
            done_flag = 0
            if done:
                done_flag = 1
                break
        t2 = time.time()

        test_result_list.append([env.current_makespan[0], t2 - t1])

    return np.array(test_result_list)


def test_sampling_strategy(data_set, model_path, sample_times, seed):
    """
        test the model on the given data using the sampling strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    test_result_list = []
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    # from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    env = MICEnvForVariousOpNums(n_j=n_j, n_m=n_m)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        # copy the testing environment
        moduleLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
        OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        t1 = time.time()
        while True:

            with torch.no_grad():
                pi, _ = ppo.policy(fea_mou=state.fea_mou_tensor.to(device),  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor.to(device),  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor.to(device),  # [sz_b, J]
                                                             fea_wor=state.fea_wor_tensor.to(device),  # [sz_b, M, 6]
                                                             worker_mask=state.worker_mask_tensor.to(device),  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor.to(device),  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor.to(device),
                                                             # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor.to(device))

            action_envs, _ = sample_action(pi)
            state, _, done = env.step(action_envs.cpu().numpy())
            if done.all():
                break

        t2 = time.time()
        best_makespan = np.min(env.current_makespan)
        test_result_list.append([best_makespan, t2 - t1])

    return np.array(test_result_list)


def main(config, flag_sample):
    """
        test the trained model following the config and save the results
    :param flag_sample: whether using the sampling strategy
    """
    setup_seed(config.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))

    # collect the test data
    test_data = pack_data_from_config(config.data_source, config.test_data)

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        save_direc = f'./test_results/{config.data_source}/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            save_path = save_direc + f'/Result_+{model[1]}_{data[1]}.npy'
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")

                all_results = []

                if not flag_sample:
                    print("Test mode: Greedy")
                    result_5_times = []
                    # Greedy mode, test 5 times, record average time.
                    for j in range(1):
                        result = test_greedy_strategy(data[0], model[0], config.seed_test)
                        result_5_times.append(result)
                    all_results = np.array(all_results)  # (num_runs, num_instances, 2)

                    # ---------- 保存完整 instance-level 结果 ----------
                    save_path = save_direc + f'/Result_{data[1]}_all.npy'
                    np.save(save_path, all_results)

                    # ---------- 仍然保留你现在的“平均结果” ----------
                    mean_result = np.mean(all_results, axis=0)  # (num_instances, 2)
                    mean_save_path = save_direc + f'/Result_{data[1]}_mean.npy'
                    np.save(mean_save_path, mean_result)

                    print(f"testing results :")
                    print(f"makespan (instance mean): ", mean_result[:, 0].mean())
                    print(f"time: ", mean_result[:, 1].mean())

                else:
                    # Sample mode, test once.
                    print("Test mode: Sample")
                    save_result = test_sampling_strategy(data[0], model[0], config.sample_times, config.seed_test)
                    print("testing results:")
                    print(f"makespan(sampling): ", save_result[:, 0].mean())
                    print(f"time: ", save_result[:, 1].mean())
                np.save(save_path, save_result)


if __name__ == '__main__':
    main(configs, False)
    # main(configs, True)
