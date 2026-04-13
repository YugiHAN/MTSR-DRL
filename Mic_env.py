from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import copy
from params import configs
import sys
import torch


@dataclass
class EnvState:
    """
        state definition
    """
    fea_mou_tensor: torch.Tensor = None
    op_mask_tensor: torch.Tensor = None
    fea_wor_tensor: torch.Tensor = None
    worker_mask_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None
    comp_idx_tensor: torch.Tensor = None
    candidate_tensor: torch.Tensor = None
    fea_pairs_tensor: torch.Tensor = None

    worker_fatigue_time: torch.Tensor = None

    worker_memory_time: torch.Tensor = None

    pair_free_time: torch.Tensor = None

    minus: torch.Tensor = None

    candidate_pt: torch.Tensor = None

    op_match_module_remain_work: torch.Tensor = None

    device = torch.device(configs.device)

    def update(self, fea_mou, op_mask, fea_wor, worker_mask, dynamic_pair_mask,
               comp_idx, candidate, fea_pairs, worker_fatigue_time, worker_memory_time, pair_free_time, minus, candidate_pt, op_match_module_remain_work):
        """
            update the state information
        """
        device = self.device
        self.fea_worou_tensor = torch.from_numpy(np.copy(fea_mou)).float().to(device)
        self.fea_wor_tensor = torch.from_numpy(np.copy(fea_wor)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.op_mask_tensor = torch.from_numpy(np.copy(op_mask)).to(device)
        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.worker_mask_tensor = torch.from_numpy(np.copy(worker_mask)).float().to(device)
        self.comp_idx_tensor = torch.from_numpy(np.copy(comp_idx)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)
        self.worker_fatigue_time_tensor = torch.from_numpy(np.copy(worker_fatigue_time)).to(device)
        self.worker_memory_time_tensor = torch.from_numpy(np.copy(worker_memory_time)).to(device)
        self.pair_free_time_tensor = torch.from_numpy(np.copy(pair_free_time)).to(device)
        self.minus_tensor = torch.from_numpy(minus).to(device)
        self.candidate_pt = torch.from_numpy(candidate_pt).to(device)
        self.op_match_module_remain_work = torch.from_numpy(op_match_module_remain_work).to(device)


    def print_shape(self):
        print(self.fea_worou_tensor.shape)
        print(self.op_mask_tensor.shape)
        print(self.candidate_tensor.shape)
        print(self.fea_wor_tensor.shape)
        print(self.worker_mask_tensor.shape)
        print(self.comp_idx_tensor.shape)
        print(self.dynamic_pair_mask_tensor.shape)
        print(self.fea_pairs_tensor.shape)




class MICEnvForVariousOpNums:
    """
        a batch of MIC fit-out scheduling environments that have various number of operations
        Attributes:

    """

    def __init__(self, n_j, n_m):
        self.number_of_modules = n_j
        self.number_of_workers = n_m
        self.old_state = EnvState()

        self.op_fea_dim = 10
        self.worker_fea_dim = 8

    def set_static_properties(self):  # 静态属性
        """
            define static properties
        """
        self.multi_env_worker_diag = np.tile(np.expand_dims(np.eye(self.number_of_workers, dtype=bool), axis=0),
                                          (self.number_of_envs, 1, 1))

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_module_idx = self.env_idxs.repeat(self.number_of_modules).reshape(self.number_of_envs, self.number_of_modules)

        # [E, N]
        self.mask_dummy_node = np.full(shape=[self.number_of_envs, self.max_number_of_ops],
                                       fill_value=False, dtype=bool)

        cols = np.arange(self.max_number_of_ops)
        self.mask_dummy_node[cols >= self.env_number_of_ops[:, None]] = True

        a = self.mask_dummy_node[:, :, np.newaxis]
        self.dummy_mask_fea_j = np.tile(a, (1, 1, self.op_fea_dim))

        self.flag_exist_dummy_node = ~(self.env_number_of_ops == self.max_number_of_ops).all()

    def set_initial_data(self, module_length_list, op_pt_list):
        self.number_of_envs = len(module_length_list)
        self.module_length = np.array(module_length_list)
        self.number_of_workers = op_pt_list[0].shape[1]
        self.number_of_modules = module_length_list[0].shape[0]

        # 异工序数环境并行化
        self.env_number_of_ops = np.array([op_pt_list[k].shape[0] for k in range(self.number_of_envs)])
        self.max_number_of_ops = np.max(self.env_number_of_ops)

        self.set_static_properties()

        self.virtual_module_length = np.copy(self.module_length)
        self.virtual_module_length[:, -1] += self.max_number_of_ops - self.env_number_of_ops

        # [E, N, M]
        self.op_pt = np.array([np.pad(op_pt_list[k],
                                      ((0, self.max_number_of_ops - self.env_number_of_ops[k]),
                                       (0, 0)),
                                      'constant', constant_values=0)
                               for k in range(self.number_of_envs)]).astype(np.float64)

        self.op_pt_temporary = np.array([np.pad(op_pt_list[k],
                                                ((0, self.max_number_of_ops - self.env_number_of_ops[k]),
                                                 (0, 0)),
                                                'constant', constant_values=0)
                                         for k in range(self.number_of_envs)]).astype(np.float64)

        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)

        self.op_pt = (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        self.process_relation = (self.op_pt != 0)
        self.reverse_process_relation = ~self.process_relation

        self.compatible_op = np.sum(self.process_relation, 2)
        self.compatible_worker = np.sum(self.process_relation, 1)

        self.unmasked_op_pt = np.copy(self.op_pt)

        head_op_id = np.zeros((self.number_of_envs, 1))

        self.module_first_op_id = np.concatenate([head_op_id, np.cumsum(self.module_length, axis=1)[:, :-1]], axis=1).astype(
            'int')
        self.module_last_op_id = self.module_first_op_id + self.module_length - 1
        self.module_last_op_id[:, -1] = self.env_number_of_ops - 1

        self.initial_vars()
        self.init_op_mask()

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data

        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt

        self.worker_min_pt = np.max(self.op_pt, axis=1).data
        self.worker_max_pt = np.max(self.op_pt, axis=1)

        self.op_ct_lb = copy.deepcopy(self.op_min_pt)

        for k in range(self.number_of_envs):
            for i in range(self.number_of_modules):
                self.op_ct_lb[k][self.module_first_op_id[k][i]:self.module_last_op_id[k][i] + 1] = np.cumsum(
                    self.op_ct_lb[k][self.module_first_op_id[k][i]:self.module_last_op_id[k][i] + 1])

        self.op_match_module_left_op_nums = np.array([np.repeat(self.module_length[k],
                                                             repeats=self.virtual_module_length[k])
                                                   for k in range(self.number_of_envs)])
        self.module_remain_work = []
        for k in range(self.number_of_envs):
            self.module_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.module_first_op_id[k][i]:self.module_last_op_id[k][i] + 1])
                 for i in range(self.number_of_modules)])

        self.op_match_module_remain_work = np.array([np.repeat(self.module_remain_work[k], repeats=self.virtual_module_length[k])
                                                  for k in range(self.number_of_envs)])

        self.construct_op_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)

        self.max_endTime = self.init_quality
        # old
        self.worker_available_op_nums = np.copy(self.compatible_worker)
        self.worker_current_available_op_nums = np.copy(self.compatible_worker)
        self.candidate_pt = np.array([self.unmasked_op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])

        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.worker_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1)

        self.worker_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        # construct worker features [E, M, 6]

        # construct Compete Tensor : [E, M, M, J]
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        # construct worker graph adjacency matrix : [E, M, M]
        self.init_worker_mask()
        self.construct_worker_features()

        self.construct_pair_features()

        self.sum_fatigue = 0

        self.old_state.update(self.fea_mou, self.op_mask,
                              self.fea_wor, self.worker_mask,
                              self.dynamic_pair_mask, self.comp_idx, self.candidate,
                              self.fea_pairs, self.worker_fatigue_time1, self.worker_memory_time, self.pair_free_time, self.module_last_op_id - self.candidate, self.candidate_pt, self.op_match_module_remain_work)

        # old record
        self.old_op_mask = np.copy(self.op_mask)
        self.old_worker_mask = np.copy(self.worker_mask)
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_match_module_left_op_nums = np.copy(self.op_match_module_left_op_nums)
        self.old_op_match_module_remain_work = np.copy(self.op_match_module_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        # self.old_pairMessage = np.copy(self.pairMessage)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_worker_current_available_op_nums = np.copy(self.worker_current_available_op_nums)
        self.old_worker_current_available_jc_nums = np.copy(self.worker_current_available_jc_nums)
        # state
        self.state = copy.deepcopy(self.old_state)
        self.memory_cache = []
        return self.state

    def reset(self):
        self.initial_vars()
        self.op_mask = np.copy(self.old_op_mask)
        self.worker_mask = np.copy(self.old_worker_mask)
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_match_module_left_op_nums = np.copy(self.old_op_match_module_left_op_nums)
        self.op_match_module_remain_work = np.copy(self.old_op_match_module_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.worker_current_available_op_nums = np.copy(self.old_worker_current_available_op_nums)
        self.worker_current_available_jc_nums = np.copy(self.old_worker_current_available_jc_nums)
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        self.step_count = 0

        self.memory_length = 4

        self.done_flag = np.full(shape=(self.number_of_envs,), fill_value=0, dtype=bool)
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        self.worker_queue = np.full(shape=[self.number_of_envs, self.number_of_workers,
                                        self.max_number_of_ops + 1], fill_value=-99, dtype=int)
        self.worker_queue_len = np.zeros((self.number_of_envs, self.number_of_workers), dtype=int)
        self.worker_queue_last_op_id = np.zeros((self.number_of_envs, self.number_of_workers), dtype=int)
        self.op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.worker_free_time = np.zeros((self.number_of_envs, self.number_of_workers))
        self.worker_remain_work = np.zeros((self.number_of_envs, self.number_of_workers))

        self.worker_waiting_time = np.zeros((self.number_of_envs, self.number_of_workers))
        self.worker_working_flag = np.zeros((self.number_of_envs, self.number_of_workers))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_modules))

        self.true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_modules))
        self.true_worker_free_time = np.zeros((self.number_of_envs, self.number_of_workers))

        self.worker_running_time = np.zeros((self.number_of_envs, self.number_of_modules, self.number_of_workers))
        self.worker_resting_time = np.zeros((self.number_of_envs, self.number_of_modules, self.number_of_workers))
        self.worker_fatigue_time = np.zeros((self.number_of_envs, self.number_of_workers, 5, 2))

        self.worker_running_time1 = np.zeros((self.number_of_envs, self.number_of_workers))
        self.worker_resting_time1 = np.zeros((self.number_of_envs, self.number_of_workers))
        self.worker_fatigue_time1 = np.zeros((self.number_of_envs, self.number_of_modules, self.number_of_workers, 5, 2))

        self.worker_memory_time = np.zeros((self.number_of_envs, self.number_of_modules, self.number_of_workers, 2))

        self.worker_chosen_flag = np.zeros((self.number_of_envs, self.number_of_workers), dtype=np.int64)
        self.worker_chosen_flag1 = np.zeros((self.number_of_envs, self.number_of_workers), dtype=np.int64)

        self.fatigue_adjustment = np.ones((self.number_of_envs, self.number_of_modules, self.number_of_workers))

        # self.fatigue_adjustment = np.random.uniform(1, 2, size=(self.number_of_envs, self.number_of_modules, self.number_of_workers))

        self.fatigue_adjustment1 = 1

        # --------------------------------

        self.candidate = np.copy(self.module_first_op_id)

        self.unscheduled_op_nums = np.copy(self.env_number_of_ops)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_modules), fill_value=0, dtype=bool)

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.op_available_worker_nums = np.copy(self.compatible_op) / self.number_of_workers
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_modules,
                                        self.number_of_workers))
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_j = np.full(shape=(self.number_of_envs, self.max_number_of_ops, self.op_fea_dim),
                                         fill_value=0, dtype=bool)
        self.sum_fatigue = 0

    def step(self, actions):
        self.incomplete_env_idx = np.where(self.done_flag == 0)[0]
        # print(self.incomplete_env_idx)
        self.number_of_incomplete_envs = int(self.number_of_envs - np.sum(self.done_flag))

        chosen_module = actions // self.number_of_workers
        chosen_worker = actions % self.number_of_workers
        chosen_op = self.candidate[self.incomplete_env_idx, chosen_module]
        

        if (self.reverse_process_relation[self.incomplete_env_idx, chosen_op, chosen_worker]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by worker {chosen_worker}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.module_last_op_id[self.incomplete_env_idx, chosen_module])
        self.candidate[self.incomplete_env_idx, chosen_module] += candidate_add_flag
        self.mask[self.incomplete_env_idx, chosen_module] = (1 - candidate_add_flag)

        self.worker_queue[
            self.incomplete_env_idx, chosen_worker, self.worker_queue_len[self.incomplete_env_idx, chosen_worker]] = chosen_op

        self.worker_queue_len[self.incomplete_env_idx, chosen_worker] += 1

        alpha = 0.1
        beta = 1.5
        gamma = 0.05
        delta = 0.1

        # [E]
        chosen_op_st = np.maximum(self.candidate_free_time[self.incomplete_env_idx, chosen_module],
                                  self.worker_free_time[self.incomplete_env_idx, chosen_worker])


        self.worker_running_time1[self.incomplete_env_idx, chosen_worker] += self.op_pt[
            self.incomplete_env_idx, chosen_op, chosen_worker
        ]

        # print(self.op_pt[self.incomplete_env_idx, chosen_op, chosen_worker].shape, self.fatigue_adjustment[self.incomplete_env_idx, chosen_module, chosen_worker].shape)

        adjusted_op_pt1 = self.op_pt[self.incomplete_env_idx, chosen_op, chosen_worker] * (
                self.fatigue_adjustment[self.incomplete_env_idx, chosen_module, chosen_worker] ** 2)

        self.op_pt_temporary = self.op_pt.copy()

        self.op_pt_temporary[self.incomplete_env_idx, chosen_op, chosen_worker] = self.op_pt[
                                                                                   self.incomplete_env_idx, chosen_op, chosen_worker] * (
                                                                                       self.fatigue_adjustment[
                                                                                           self.incomplete_env_idx, chosen_module, chosen_worker] ** 2)

        self.op_ct[self.incomplete_env_idx, chosen_op] = chosen_op_st + adjusted_op_pt1

        # print(self.op_ct[self.incomplete_env_idx, chosen_op])

        self.candidate_free_time[self.incomplete_env_idx, chosen_module] = self.op_ct[self.incomplete_env_idx, chosen_op]
        self.worker_free_time[self.incomplete_env_idx, chosen_worker] = self.op_ct[self.incomplete_env_idx, chosen_op]

        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.incomplete_env_idx, chosen_module],
                                       self.true_worker_free_time[self.incomplete_env_idx, chosen_worker])

        adjusted_op_pt = self.true_op_pt[self.incomplete_env_idx, chosen_op, chosen_worker] * (
                self.fatigue_adjustment[self.incomplete_env_idx, chosen_module, chosen_worker] ** 2)

        self.sum_fatigue += np.mean(self.fatigue_adjustment[self.incomplete_env_idx, chosen_module, chosen_worker] ** 2)

        self.fatigue = self.fatigue_adjustment[self.incomplete_env_idx, chosen_module, chosen_worker] ** 2

        self.true_op_ct[self.incomplete_env_idx, chosen_op] = true_chosen_op_st + adjusted_op_pt

        self.true_candidate_free_time[self.incomplete_env_idx, chosen_module] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]
        self.true_worker_free_time[self.incomplete_env_idx, chosen_worker] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]

        self.current_makespan[self.incomplete_env_idx] = np.maximum(self.current_makespan[self.incomplete_env_idx],
                                                                    self.true_op_ct[
                                                                        self.incomplete_env_idx, chosen_op])

        for k, j in enumerate(self.incomplete_env_idx):
            if candidate_add_flag[k]:
                # print(self.fatigue_adjustment[j, chosen_module].shape)
                self.candidate_pt[j, chosen_module[k]] = self.unmasked_op_pt[j, chosen_op[k] + 1] * (
                self.fatigue_adjustment[j, chosen_module[k]] ** 2)
                self.candidate_process_relation[j, chosen_module[k]] = self.reverse_process_relation[j, chosen_op[k] + 1]
            else:
                self.candidate_process_relation[j, chosen_module[k]] = 1

        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        workerFT_for_compare = np.expand_dims(self.worker_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, workerFT_for_compare)

        self.worker_running_time = np.repeat(np.expand_dims(self.worker_running_time1, 1), self.number_of_modules, 1)

        self.worker_resting_time = self.pair_free_time - self.worker_running_time


        for k in range(len(self.incomplete_env_idx)):
            if self.worker_chosen_flag1[k, chosen_worker[k]] > self.memory_length:
                self.worker_memory_time[k, :, :, :] = 0
                self.worker_memory_time[k, chosen_module[k], chosen_worker[k], 0] = self.worker_fatigue_time1[
                    k, chosen_module[k], chosen_worker[k], 0, 0]
                self.worker_memory_time[k, chosen_module[k], chosen_worker[k], 1] = self.worker_fatigue_time1[
                    k, chosen_module[k], chosen_worker[k], 0, 1]
                self.worker_fatigue_time1[k, chosen_module[k], chosen_worker[k], :self.memory_length, 0] = self.worker_fatigue_time1[
                                                                                     k, chosen_module[k], chosen_worker[k],
                                                                                     1:, 0]

                self.worker_fatigue_time1[k, chosen_module[k], chosen_worker[k], self.memory_length, 0] = self.worker_running_time[
                    k, chosen_module[k], chosen_worker[k]]

                self.worker_fatigue_time1[k, chosen_module[k], chosen_worker[k], :self.memory_length, 1] = self.worker_fatigue_time1[
                                                                                     k, chosen_module[k], chosen_worker[k],
                                                                                     1:, 1]
                self.worker_fatigue_time1[k, chosen_module[k], chosen_worker[k], self.memory_length, 1] = self.worker_resting_time[
                    k, chosen_module[k], chosen_worker[k]]

            else:
                self.worker_fatigue_time1[
                    k, chosen_module[k], chosen_worker[k], self.worker_chosen_flag1[k, chosen_worker[k]], 0] = \
                    self.worker_running_time[k, chosen_module[k], chosen_worker[k]]
                self.worker_fatigue_time1[
                    k, chosen_module[k], chosen_worker[k], self.worker_chosen_flag1[k, chosen_worker[k]], 1] = \
                    self.worker_resting_time[k, chosen_module[k], chosen_worker[k]]

        self.worker_chosen_flag1[self.incomplete_env_idx, chosen_worker] += 1

        self.fatigue_adjustment = self.calculate_fatigue_adjustment(
            self.worker_fatigue_time1[:, :, :, :, 0] * 100,
            self.worker_fatigue_time1[:, :, :, :, 1] * 100, alpha, beta, gamma, delta)

        pair_free_time = self.pair_free_time[self.incomplete_env_idx]

        schedule_matrix = ma.array(pair_free_time, mask=self.candidate_process_relation[self.incomplete_env_idx])

        self.next_schedule_time[self.incomplete_env_idx] = np.min(
            schedule_matrix.reshape(self.number_of_incomplete_envs, -1), axis=1).data

        self.remain_process_relation[self.incomplete_env_idx, chosen_op] = 0
        self.op_scheduled_flag[self.incomplete_env_idx, chosen_op] = 1

        self.deleted_op_nodes = \
            np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.op_scheduled_flag)

        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                         (1, 1, self.op_fea_dim))

        self.update_op_mask()

        self.worker_queue_last_op_id[self.incomplete_env_idx, chosen_worker] = chosen_op

        self.unscheduled_op_nums[self.incomplete_env_idx] -= 1

        diff = self.op_ct[self.incomplete_env_idx, chosen_op] - self.op_ct_lb[self.incomplete_env_idx, chosen_op]
        for k, j in enumerate(self.incomplete_env_idx):
            self.op_ct_lb[j][chosen_op[k]:self.module_last_op_id[j, chosen_module[k]] + 1] += diff[k]
            self.op_match_module_left_op_nums[j][
            self.module_first_op_id[j, chosen_module[k]]:self.module_last_op_id[j, chosen_module[k]] + 1] -= 1
            self.op_match_module_remain_work[j][
            self.module_first_op_id[j, chosen_module[k]]:self.module_last_op_id[j, chosen_module[k]] + 1] -= \
                self.op_mean_pt[j, chosen_op[k]]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time[self.env_module_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_module_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)


        self.construct_op_features()

        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = np.array([pair_free_time[k] > self.next_schedule_time[j]
                                           for k, j in enumerate(self.incomplete_env_idx)])
        self.dynamic_pair_mask[self.incomplete_env_idx] = np.logical_or(self.dynamic_pair_mask[self.incomplete_env_idx],
                                                                        self.unavailable_pairs)

        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)

        self.update_worker_mask()

        self.worker_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.worker_current_available_op_nums[self.incomplete_env_idx] -= self.process_relation[
            self.incomplete_env_idx, chosen_op]

        worker_free_duration = np.expand_dims(self.next_schedule_time[self.
                                           incomplete_env_idx], axis=1) - self.worker_free_time[self.incomplete_env_idx]
        worker_free_flag = worker_free_duration < 0
        self.worker_working_flag[self.incomplete_env_idx] = worker_free_flag + 0
        self.worker_waiting_time[self.incomplete_env_idx] = (1 - worker_free_flag) * worker_free_duration

        self.worker_remain_work[self.incomplete_env_idx] = np.maximum(-worker_free_duration, 0)

        self.construct_worker_features()

        self.construct_pair_features()

        reward = self.max_endTime - np.max(self.op_ct_lb,
                                           axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        self.state.update(self.fea_mou, self.op_mask, self.fea_wor, self.worker_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs, self.worker_fatigue_time1, self.worker_memory_time, self.pair_free_time, self.module_last_op_id - self.candidate, self.candidate_pt, self.op_match_module_remain_work)
        self.done_flag = self.done()

        return self.state, np.array(reward), self.done_flag, self.fatigue

    def done(self):
        return self.step_count >= self.env_number_of_ops

    def construct_op_features(self):

        self.fea_mou = np.stack((self.op_scheduled_flag,
                               self.op_ct_lb,
                               self.op_min_pt,
                               self.pt_span,
                               self.op_mean_pt,
                               self.op_waiting_time,
                               self.op_remain_work,
                               self.op_match_module_left_op_nums,
                               self.op_match_module_remain_work,
                               self.op_available_worker_nums), axis=2)

        if self.flag_exist_dummy_node:
            mask_all = np.logical_or(self.dummy_mask_fea_j, self.delete_mask_fea_j)
        else:
            mask_all = self.delete_mask_fea_j

        self.norm_operation_features(mask=mask_all)

    def norm_operation_features(self, mask):
        self.fea_mou[mask] = 0
        num_delete_nodes = np.count_nonzero(mask[:, :, 0], axis=1)

        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = self.max_number_of_ops - num_delete_nodes

        num_left_nodes = np.maximum(num_left_nodes, 1e-8)

        mean_fea_j = np.sum(self.fea_mou, axis=1) / num_left_nodes

        temp = np.where(self.delete_mask_fea_j,
                        mean_fea_j[:, np.newaxis, :], self.fea_mou)
        var_fea_j = np.var(temp, axis=1)

        std_fea_j = np.sqrt(var_fea_j * self.max_number_of_ops / num_left_nodes)

        self.fea_mou = ((temp - mean_fea_j[:, np.newaxis, :]) / \
                      (std_fea_j[:, np.newaxis, :] + 1e-8))

    def construct_worker_features(self):

        self.fea_wor = np.stack((self.worker_current_available_jc_nums,
                               self.worker_current_available_op_nums,
                               self.worker_min_pt,
                               self.worker_mean_pt,
                               self.worker_waiting_time,
                               self.worker_remain_work,
                               self.worker_free_time,
                               self.worker_working_flag), axis=2)

        self.norm_worker_features()

    '''
    def calculate_fatigue_adjustment(self,worker_running_time,worker_resting_time, alpha, beta, gamma, delta):
        fatigue = alpha * (worker_running_time ** beta) - gamma * np.exp(-delta * worker_resting_time)

        fatigue = np.maximum(fatigue, 0)
        fatigue_scaled = 1 + 0.001 * fatigue
        return fatigue_scaled
    '''

    def calculate_fatigue_adjustment(self, worker_running_time, worker_resting_time, alpha, beta, gamma, delta):

        # print(1, worker_running_time[0, 0, :], 2, worker_resting_time[0, 0, :])

        # print(111, alpha * (worker_running_time[0, 0, :] ** beta), 222, 10 * alpha * (worker_resting_time[0, 0, :] ** beta))

        # fatigue = alpha * (np.sum(worker_running_time, axis=1) ** beta) - gamma * np.exp(-delta * np.sum(worker_resting_time, axis=1))

        #         fatigue = alpha * (np.sum(worker_running_time, axis=1) ** beta) - alpha * (np.sum(worker_resting_time, axis=1) ** beta)

        #         fatigue = np.maximum(fatigue, 0)
        #         fatigue_scaled = 1 + 0.0001 * fatigue

        fatigue = alpha * (np.sum(worker_running_time ** beta, axis=3)) - alpha * (
            np.sum(worker_resting_time ** beta, axis=3))
        # fatigue = alpha * (worker_running_time ** beta) - alpha * (worker_resting_time ** beta)

        fatigue = np.maximum(fatigue, 0)
        fatigue_scaled = 1 + 0.001 * fatigue

        # print(fatigue_scaled[0, 0, :])

        return fatigue_scaled

    # def calculate_fatigue_adjustment(self, worker_running_time, worker_resting_time, alpha, beta, gamma, delta):
    #     """
    #     计算疲劳调整值，加入动态随机因素和非线性影响
    #     :param worker_running_time: 机器当前运行时间矩阵
    #     :param worker_resting_time: 机器当前休息时间矩阵
    #     :param alpha: 控制疲劳增长速率
    #     :param beta: 控制疲劳曲线的非线性程度
    #     :param gamma: 调整随机扰动幅度
    #     :param delta: 调整非线性因子的比例
    #     :return: 计算后的疲劳度调整值
    #     """
    #     # 基础疲劳度计算
    #     base_fatigue = alpha * (worker_running_time ** beta) - alpha * (worker_resting_time ** beta)
    #
    #     # 动态非线性疲劳调整
    #     interaction_effect = delta * np.log1p(worker_running_time * worker_resting_time)
    #
    #     # 引入随机扰动（基于当前状态调整）
    #     random_state = np.random.normal(loc=1.0, scale=gamma, size=worker_running_time.shape)
    #     random_state = np.clip(random_state, 0.8, 1.2)  # 限制随机扰动幅度
    #
    #     # 综合疲劳度计算
    #     fatigue = base_fatigue + interaction_effect * random_state
    #
    #     # 确保疲劳度非负，并进行缩放
    #     fatigue = np.maximum(fatigue, 0)
    #     fatigue_scaled = 1 + 0.005 * fatigue
    #
    #     print(fatigue_scaled[0, 0, :])  # 打印部分疲劳值用于调试
    #
    #     return fatigue_scaled

    '''

    def calculate_fatigue_adjustment(self, run_times, rest_times, alpha=0.1, beta=0.05, gamma=5):
        """
        计算基于运行时间和休息时间的疲劳调整值。
        参数:
        run_times (Tensor): 机器的运行时间序列。
        rest_times (Tensor): 机器的休息时间序列。
        返回:
        Tensor: 当前机器的疲劳值。
        """
        #print(run_times)
        # 计算疲劳累积，假设随运行时间指数增长
        fatigue_accumulation = 1 / (1 + np.exp(-alpha * run_times * gamma))

        # 疲劳恢复同样使用Logistic函数，但方向相反
        fatigue_recovery = 1 / (1 + np.exp(beta * rest_times))

        # 计算最终的疲劳值
        fatigue = np.sum(fatigue_accumulation - fatigue_recovery)

        # 将疲劳值归一化到1到2之间
        fatigue_normalized = 1 + (fatigue / np.max(fatigue))
        print(fatigue_normalized)
        return fatigue_normalized

        #return scaled_fatigue.cpu().numpy()
    '''

    def norm_worker_features(self):
        self.fea_wor[self.delete_mask_fea_m] = 0
        num_delete_workers = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_workers = num_delete_workers[:, np.newaxis]
        num_left_workers = self.number_of_workers - num_delete_workers

        num_left_workers = np.maximum(num_left_workers, 1e-8)

        mean_fea_m = np.sum(self.fea_wor, axis=1) / num_left_workers

        temp = np.where(self.delete_mask_fea_m,
                        mean_fea_m[:, np.newaxis, :], self.fea_wor)
        var_fea_m = np.var(temp, axis=1)

        std_fea_m = np.sqrt(var_fea_m * self.number_of_workers / num_left_workers)

        self.fea_wor = ((temp - mean_fea_m[:, np.newaxis, :]) / \
                      (std_fea_m[:, np.newaxis, :] + 1e-8))

    def construct_pair_features(self):

        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_module_idx, self.candidate], axis=-1)

        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True) \
            .filled(0 + 1e-8)

        worker_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True). \
            filled(0 + 1e-8)

        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True),
                             axis=2, keepdims=True) + 1e-8

        worker_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        pair_wait_time = self.op_waiting_time[self.env_module_idx, self.candidate][:, :,
                         np.newaxis] + self.worker_waiting_time[:, np.newaxis, :]

        chosen_module_remain_work = np.expand_dims(self.op_match_module_remain_work
                                                [self.env_module_idx, self.candidate],
                                                axis=-1) + 1e-8

        self.fea_pairs = np.stack((self.candidate_pt,
                                   self.candidate_pt / chosen_op_max_pt,
                                   self.candidate_pt / worker_max_candidate_pt,
                                   self.candidate_pt / max_remain_op_pt,
                                   self.candidate_pt / worker_max_remain_op_pt,
                                   self.candidate_pt / pair_max_pt,
                                   self.candidate_pt / chosen_module_remain_work,
                                   pair_wait_time), axis=-1)

    def update_worker_mask(self):

        self.worker_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.worker_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.worker_fea_dim))
        self.worker_mask[self.multi_env_worker_diag] = 1

    def init_worker_mask(self):

        self.worker_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.worker_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.worker_fea_dim))
        self.worker_mask[self.multi_env_worker_diag] = 1

    def init_op_mask(self):
        self.op_mask = np.full(shape=(self.number_of_envs, self.max_number_of_ops, 3),
                               fill_value=0, dtype=np.float32)
        self.op_mask[self.env_module_idx, self.module_first_op_id, 0] = 1
        self.op_mask[self.env_module_idx, self.module_last_op_id, 2] = 1

    def update_op_mask(self):
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)

    def logic_operator(self, x, flagT=True):
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)

        return np.logical_and(d1, d2).astype(np.float32)


'''


        greater_than_five_indices = np.where(self.worker_chosen_flag[self.incomplete_env_idx, chosen_worker] >=5)

        print(greater_than_five_indices)

        below_five_indices = np.where(self.worker_chosen_flag[self.incomplete_env_idx, chosen_worker] < 5)

        #print(self.worker_fatigue_time[greater_than_five_indices, chosen_worker, :4, 0].shape)


        if greater_than_five_indices[0].size > 0:
            print(self.worker_fatigue_time[greater_than_five_indices[0], greater_than_five_indices[1], :4, 0].shape)
            self.worker_fatigue_time[greater_than_five_indices[0], greater_than_five_indices[1], :4, 0] = self.worker_fatigue_time[
                                                                                    greater_than_five_indices[0], greater_than_five_indices[1], 1:,
                                                                                    0]
            self.worker_fatigue_time[greater_than_five_indices[0], greater_than_five_indices[1], 4, 0] = self.worker_running_time[
                greater_than_five_indices[0], greater_than_five_indices[1]]

        if below_five_indices[0].size > 0:
            for i in range(below_five_indices[0].size):
                self.worker_fatigue_time[
                    below_five_indices[0], below_five_indices[1], self.worker_chosen_flag[below_five_indices[0], below_five_indices[1]], 0] = \
                self.worker_running_time[below_five_indices[0], below_five_indices[1]]





        if self.worker_chosen_flag[self.incomplete_env_idx, chosen_worker]>=5:
           self.worker_fatigue_time[self.incomplete_env_idx, chosen_worker, :4, 0] = self.worker_fatigue_time[self.incomplete_env_idx, chosen_worker, 1:, 0]
           self.worker_fatigue_time[self.incomplete_env_idx, chosen_worker, 4, 0] = self.worker_running_time[self.incomplete_env_idx, chosen_worker]
           #self.worker_running_sequence.append(self.worker_running_time[self.incomplete_env_idx, chosen_worker])
           #self.worker_running_sequence.pop(0)
        else:
           self.worker_fatigue_time[self.incomplete_env_idx, chosen_worker, self.worker_chosen_flag[self.incomplete_env_idx, chosen_worker], 0] = self.worker_running_time[self.incomplete_env_idx, chosen_worker]
        '''