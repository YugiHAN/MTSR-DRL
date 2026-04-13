import json
import random

from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy

"""
    agent utils
"""


def sample_action(p):
    """
        sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    _, index = torch.max(p, dim=1)
    return index


def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_worker_list_for_module(chosen_module, env):
    """
    :param chosen_module: the selected module
    :param env: the production environment
    :return: the workers which can immediately process the chosen module
    """
    worker_state = ~env.candidate_process_relation[0, chosen_module]
    available_worker_list = np.where(worker_state == True)[0]
    worker_free_time = env.worker_free_time[0][available_worker_list]
    module_free_time = env.candidate_free_time[0][chosen_module]
    # case1 eg:
    # JF: 50
    # workerF: 55 60 65 70
    if (module_free_time < worker_free_time).all():
        chosen_worker_list = available_worker_list[min_element_index(worker_free_time)]
    # case2 eg:
    # JF: 50
    # workerF: 35 40 55 60
    else:
        chosen_worker_list = available_worker_list[np.where(worker_free_time <= module_free_time)]

    return chosen_worker_list

def select_agv_min_eet(env, chosen_module, chosen_worker):
    """
    Select AGV minimizing earliest executable time.
    If no AGV is available at current schedule time,
    fallback to globally earliest-free AGV.
    """

    # Step 1: check AGV availability under current mask
    agv_state = ~env.agv_working_mask[0]
    available_agvs = np.where(agv_state == True)[0]

    # ===== 核心修复点 =====
    if available_agvs.size == 0:
        # No AGV available at current decision time
        # → select AGV with earliest free time (time-advance consistent)
        min_ft = np.min(env.agv_free_time[0])
        available_agvs = np.where(env.agv_free_time[0] == min_ft)[0]

    # Step 2: compute transport time
    module_last_worker = env.last_worker_module_list[0, chosen_module]
    agv_last_worker = env.last_worker_agv_list[0, available_agvs]

    transit_time1 = env.transit_time[0, agv_last_worker, module_last_worker]
    transit_time2 = env.transit_time[0, module_last_worker, chosen_worker]
    transit_flag = (transit_time2 > 0).astype(np.int32)
    transit_time = transit_time1 * transit_flag + transit_time2

    # Step 3: compute EET
    agv_ready = env.agv_free_time[0, available_agvs]
    module_ready = env.candidate_free_time[0, chosen_module]
    worker_ready = env.worker_free_time[0, chosen_worker]

    eet = np.maximum.reduce([agv_ready, module_ready, worker_ready]) + transit_time

    chosen_agv_list = available_agvs[min_element_index(eet)]
    return np.random.choice(chosen_agv_list)



def heuristic_select_action(method, env):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    chosen_module = -1
    chosen_worker = -1

    module_state = (env.mask[0] == 0)

    process_module_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    module_state = process_module_state & module_state

    available_modules = np.where(module_state == True)[0]
    available_ops = env.candidate[0][available_modules]

    if method == 'MOPNR+SPT':
        # MOPNR: Selecting the module with the maximum number of operations remaining
        remain_ops = env.op_match_module_left_op_nums[0][available_ops]
        chosen_module_list = available_modules[max_element_index(remain_ops)]
        chosen_module = np.random.choice(chosen_module_list)

        # SPT: Selecting the worker with the shortest processing time for the chosen module
        worker_state = ~env.candidate_process_relation[0, chosen_module]
        available_workers = np.where(worker_state == True)[0]
        worker_pt = env.candidate_pt[0, chosen_module, available_workers]
        chosen_worker_list = available_workers[min_element_index(worker_pt)]
        chosen_worker = np.random.choice(chosen_worker_list)

    elif method == 'MWKR+SPT':
        # MWKR: Selecting the module with the maximum remaining work
        module_remain_work_list = env.op_match_module_remain_work[0][available_ops]
        chosen_module_list = available_modules[max_element_index(module_remain_work_list)]
        chosen_module = np.random.choice(chosen_module_list)
        # chosen_module = chosen_module_list[0]

        # SPT: Selecting the worker with the shortest processing time for the chosen module
        worker_state = ~env.candidate_process_relation[0, chosen_module]
        available_workers = np.where(worker_state == True)[0]
        worker_pt = env.candidate_pt[0, chosen_module, available_workers]
        # print(worker_pt)
        chosen_worker_list = available_workers[min_element_index(worker_pt)]
        chosen_worker = np.random.choice(chosen_worker_list)

    elif method == 'FIFO+EET':
        # FIFO: Select the module with the earliest ready time
        candidate_free_time = env.candidate_free_time[0][available_modules]
        chosen_module_list = available_modules[min_element_index(candidate_free_time)]
        chosen_module = np.random.choice(chosen_module_list)

        # EET: Select the worker with the earliest end time for the chosen module
        worker_state = ~env.candidate_process_relation[0, chosen_module]  # workers that can process the module
        available_workers = np.where(worker_state == True)[0]

        # Calculate end times for all available workers
        worker_free_time = env.worker_free_time[0][available_workers]  # worker free times
        worker_processing_time = env.candidate_pt[0, chosen_module, available_workers]  # Processing times for the module
        worker_end_time = worker_free_time + worker_processing_time  # End time = free time + processing time

        chosen_worker_list = available_workers[min_element_index(worker_end_time)]
        chosen_worker = np.random.choice(chosen_worker_list)

    elif method == 'FIFO+SPT':
        # Step 1: FIFO - Select the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_modules]
        min_time_index = min_element_index(candidate_free_time)
        chosen_module_list = available_modules[min_time_index]
        chosen_module = np.random.choice(chosen_module_list)

        # Step 2: SPT - Select the worker with the shortest processing time for the chosen module
        # Get all available workers for the chosen module
        worker_state = ~env.candidate_process_relation[0, chosen_module]
        available_workers = np.where(worker_state == True)[0]

        # Retrieve processing times for the chosen module on the available workers
        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt[chosen_module, available_workers]

        # Select the worker with the shortest processing time
        chosen_worker_list = available_workers[np.where(pt_list == np.min(pt_list))[0]]
        chosen_worker = np.random.choice(chosen_worker_list)

    elif method == 'MOPNR+EET':
        # MOPNR: Select the module with the maximum number of remaining operations
        remain_ops = env.op_match_module_left_op_nums[0][available_ops]
        chosen_module_list = available_modules[max_element_index(remain_ops)]
        chosen_module = np.random.choice(chosen_module_list)

        # EET: Select the worker with the earliest end time for the chosen module
        worker_state = ~env.candidate_process_relation[0, chosen_module]  # workers that can process the module
        available_workers = np.where(worker_state == True)[0]

        # Calculate end times for all available workers
        worker_free_time = env.worker_free_time[0][available_workers]  # worker free times
        worker_processing_time = env.candidate_pt[0, chosen_module, available_workers]  # Processing times for the module
        worker_end_time = worker_free_time + worker_processing_time  # End time = free time + processing time

        chosen_worker_list = available_workers[min_element_index(worker_end_time)]
        chosen_worker = np.random.choice(chosen_worker_list)

    elif method == 'MWKR+EET':
        # MWKR: Select the module with the maximum remaining work
        module_remain_work_list = env.op_match_module_remain_work[0][available_ops]
        chosen_module_list = available_modules[max_element_index(module_remain_work_list)]
        chosen_module = np.random.choice(chosen_module_list)

        # EET: Select the worker with the earliest end time for the chosen module
        worker_state = ~env.candidate_process_relation[0, chosen_module]  # workers that can process the module
        available_workers = np.where(worker_state == True)[0]

        # Calculate end times for all available workers
        worker_free_time = env.worker_free_time[0][available_workers]  # worker free times
        worker_processing_time = env.candidate_pt[0, chosen_module, available_workers]  # Processing times for the module
        worker_end_time = worker_free_time + worker_processing_time  # End time = free time + processing time

        chosen_worker_list = available_workers[min_element_index(worker_end_time)]
        chosen_worker = np.random.choice(chosen_worker_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_module == -1 or chosen_worker == -1:
        print(f'Error From choosing action: choose module {chosen_module}, worker {chosen_worker}')
        sys.exit()

    action = chosen_module * env.number_of_workers + chosen_worker
    return action


"""
    common utils
"""


def save_default_params(config):
    """
        save parameters in the config
    :param config: a package of parameters
    :return:
    """
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)
    print("successfully save default params")


def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('123')
