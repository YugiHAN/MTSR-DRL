from ortools.sat.python import cp_model
import numpy as np
import time
import os
from tqdm import tqdm
import sys
from params import configs
from data_utils import pack_data_from_config
import collections

import gurobipy as gp
from gurobipy import Model, GRB, quicksum

from Mic_env import MICEnvForVariousOpNums

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


def compute_earliest_start(modules):
    stime = {}
    idx = 0
    for j, module in enumerate(modules):
        prev = 0
        for t, task in enumerate(module):
            min_pt = min(pt for pt, _ in task)
            stime[idx] = prev
            prev += min_pt
            idx += 1
    return stime

def solve_instances(config):
    """
        Solve 'test_data' from 'data_source' using OR-Tools
        with time limits 'max_solve_time' for each instance,
        and save the result to './or_solution/{data_source}'
    :param config: a package of parameters
    :return:
    """
    # p = psutil.Process()
    # p.cpu_affinity(range(config.low, config.high))

    if not os.path.exists(f'./or_solution/{config.data_source}'):
        os.makedirs(f'./or_solution/{config.data_source}')

    data_list = pack_data_from_config(config.data_source, config.test_data)

    save_direc = f'./or_solution/{config.data_source}'
    if not os.path.exists(save_direc):
        os.makedirs(save_direc)

    for data in data_list:
        dataset = data[0]
        data_name = data[1]
        save_path = save_direc + f'/solution_{data_name}.npy'
        save_subpath = save_direc + f'/{data_name}'

        if not os.path.exists(save_subpath):
            os.makedirs(save_subpath)

        if (not os.path.exists(save_path)) or config.cover_flag:
            print("-" * 25 + "Solve Setting" + "-" * 25)
            print(f"solve data name : {data_name}")
            print(f"path : ./data/{config.data_source}/{data_name}")

            # search for the start index
            for root, dirs, files in os.walk(save_subpath):
                index = len([int(f.split("_")[-1][:-4]) for f in files])

            print(f"left instances: dataset[{index}, {len(dataset[0])})")

            result = []

            for i in tqdm(range(index, len(dataset[0])), file=sys.stdout, desc="progress", colour='blue'):
                n_j = dataset[0][i].shape[0]
                n_op, n_m = dataset[1][i].shape

                # 1)求解
                modules, num_workers = matrix_to_the_format_for_solving(dataset[0][i], dataset[1][i])
                static_schedule, static_obj, solve_time = fjsp_solver_ortools(
                    modules=modules,
                    num_workers=num_workers,
                    time_limits=config.max_solve_time
                )

                if static_schedule is None:
                    result.append([np.inf, solve_time])
                    continue

                # 2)动态环境执行
                env = MICEnvForVariousOpNums(n_j=n_j, n_m=n_m)
                env.set_initial_data([dataset[0][i]], [dataset[1][i]])

                t1 = time.time()
                scheduled_flags = [False] * len(static_schedule)

                while True:
                    action = ortools_select_action_from_schedule(env, static_schedule, scheduled_flags)

                    if action is None:
                        raise RuntimeError("No feasible OR-Tools action found in current dynamic state.")

                    _, _, done, _ = env.step(np.array([action]))

                    if done:
                        break

                t2 = time.time()

                print(env.current_makespan[0])

                result.append([env.current_makespan[0], solve_time + (t2 - t1)])

            return np.array(result)


def matrix_to_the_format_for_solving(module_length, op_pt):
    """
        Convert matrix form of the data into the format needed by OR-Tools
    :param module_length: the number of operations in each module (shape [J])
    :param op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth worker or 0 if $O_i$ can not process on $M_j$
    :return:
    """
    num_ops, num_workers = op_pt.shape
    num_modules = module_length.shape[0]
    modules = []
    op_idx = 0
    for j in range(num_modules):
        module_msg = []
        for k in range(module_length[j]):
            able_mchs = np.where(op_pt[op_idx] != 0)[0]
            op_msg = [(op_pt[op_idx, k], k) for k in able_mchs]
            module_msg.append(op_msg)
            op_idx += 1
        modules.append(module_msg)
    return modules, num_workers


def ortools_select_action_from_schedule(env, static_schedule, scheduled_flags):
    """
    Select the earliest unscheduled and currently feasible action
    according to the OR-Tools static schedule.
    """
    for idx, item in enumerate(static_schedule):
        if scheduled_flags[idx]:
            continue

        chosen_module = item["module_id"]
        chosen_worker = item["worker_id"]

        action = chosen_module * env.number_of_workers + chosen_worker

        scheduled_flags[idx] = True
        return action

    return None


def fjsp_solver_ortools(modules, num_workers, time_limits):
    """
            solve a fjsp instance by OR-Tools
            (imported from https://github.com/google/or-tools/blob/master/examples/python/flexible_module_shop_sat.py)
        :param modules: a list of processing information
            eg. modules = [  # task = (processing_time, worker_id)
                            [  # module 0
                                [(3, 0), (1, 1), (5, 2)],  # task 0 with 3 alternatives
                                [(2, 0), (4, 1), (6, 2)],  # task 1 with 3 alternatives
                                [(2, 0), (3, 1), (1, 2)],  # task 2 with 3 alternatives
                            ],
                            [  # module 1
                                [(2, 0), (3, 1), (4, 2)],
                                [(1, 0), (5, 1), (4, 2)],
                                [(2, 0), (1, 1), (4, 2)],
                            ],
                            [  # module 2
                                [(2, 0), (1, 1), (4, 2)],
                                [(2, 0), (3, 1), (4, 2)],
                                [(3, 0), (1, 1), (5, 2)],
                            ],
                        ]
        :param num_workers: the number of workers
        :param time_limits: the time limits for solving the instance
    """
    num_modules = len(modules)
    all_modules = range(num_modules)
    all_workers = range(num_workers)

    model = cp_model.CpModel()

    horizon = 0
    for module in modules:
        for task in module:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration

    intervals_per_resources = collections.defaultdict(list)
    starts = {}
    ends = {}
    presences = {}
    module_ends = []

    for module_id in all_modules:
        module = modules[module_id]
        num_tasks = len(module)
        previous_end = None
        for task_id in range(num_tasks):
            task = module[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]
            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            suffix_name = f'_j{module_id}_t{task_id}'
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration, 'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end, 'interval' + suffix_name)

            starts[(module_id, task_id)] = start
            ends[(module_id, task_id)] = end

            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = f'_j{module_id}_t{task_id}_a{alt_id}'
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence, 'interval' + alt_suffix
                    )
                    l_presences.append(l_presence)

                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    worker_id = task[alt_id][1]
                    intervals_per_resources[worker_id].append(l_interval)
                    presences[(module_id, task_id, alt_id)] = l_presence

                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(module_id, task_id, 0)] = model.NewConstant(1)

        module_ends.append(previous_end)

    for worker_id in all_workers:
        intervals = intervals_per_resources[worker_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, module_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limits

    total1 = time.time()
    status = solver.Solve(model)
    total2 = time.time()

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, float("inf"), total2 - total1

    schedule = []
    for module_id in all_modules:
        for task_id, task in enumerate(modules[module_id]):
            chosen_alt = None
            chosen_worker = None
            chosen_duration = None
            for alt_id in range(len(task)):
                if solver.Value(presences[(module_id, task_id, alt_id)]) == 1:
                    chosen_alt = alt_id
                    chosen_duration, chosen_worker = task[alt_id]
                    break

            schedule.append({
                "module_id": module_id,
                "task_id": task_id,
                "worker_id": chosen_worker,
                "start": solver.Value(starts[(module_id, task_id)]),
                "end": solver.Value(ends[(module_id, task_id)]),
                "duration": chosen_duration
            })

    schedule.sort(key=lambda x: (x["start"], x["module_id"], x["task_id"]))

    return schedule, solver.ObjectiveValue(), total2 - total1


def fjsp_solver_gurobi(modules, num_workers, time_limits):
    """
    Gurobi-based FJSP solver
    Data format strictly follows matrix_to_the_format_for_solving()
    """

    num_modules = len(modules)
    J = range(num_modules)

    # operations per module
    OJ = {j: range(len(modules[j])) for j in J}

    # workers & processing times
    operations_workers = {}
    operations_times = {}

    horizon = 0
    for j in J:
        for i in OJ[j]:
            ops = modules[j][i]
            operations_workers[j, i] = []
            for (pt, k) in ops:
                operations_workers[j, i].append(k)
                operations_times[j, i, k] = pt
                horizon += pt

    largeM = horizon

    # earliest start preprocessing (same logic as your MIPModel)
    stimeOp = {}
    stimeOpMax = 0
    for j in J:
        for i in OJ[j]:
            if i == 0:
                stimeOp[j, i] = 0
            else:
                prev_i = i - 1
                min_pt = min(operations_times[j, prev_i, k]
                             for k in operations_workers[j, prev_i])
                stimeOp[j, i] = stimeOp[j, prev_i] + min_pt
            stimeOpMax = max(stimeOpMax, stimeOp[j, i])

    # --------------------
    # Gurobi model
    # --------------------
    model = gp.Model("FJSP_Gurobi")

    x, y, s = {}, {}, {}

    # variables
    cmax = model.addVar(lb=stimeOpMax, ub=largeM, vtype=GRB.INTEGER, name="cmax")

    for j in J:
        for i in OJ[j]:
            s[j, i] = model.addVar(lb=stimeOp[j, i],
                                   ub=largeM,
                                   vtype=GRB.INTEGER,
                                   name=f"s({j},{i})")
            for k in operations_workers[j, i]:
                x[j, i, k] = model.addVar(vtype=GRB.BINARY,
                                          name=f"x({j},{i},{k})")

    for j in J:
        for jp in J:
            if j < jp:
                for i in OJ[j]:
                    for ip in OJ[jp]:
                        y[j, i, jp, ip] = model.addVar(vtype=GRB.BINARY,
                                                       name=f"y({j},{i},{jp},{ip})")

    model.update()

    # --------------------
    # Objective
    # --------------------
    model.setObjective(cmax, GRB.MINIMIZE)

    # --------------------
    # Constraints
    # --------------------

    # (3) assignment
    for j in J:
        for i in OJ[j]:
            model.addConstr(
                quicksum(x[j, i, k] for k in operations_workers[j, i]) == 1,
                name=f"assign({j},{i})"
            )

    # (4) precedence inside module
    for j in J:
        for i in OJ[j]:
            if i > 0:
                model.addConstr(
                    s[j, i] >= s[j, i - 1] +
                    quicksum(operations_times[j, i - 1, k] * x[j, i - 1, k]
                             for k in operations_workers[j, i - 1]),
                    name=f"prec({j},{i})"
                )

    # (5) worker non-overlap (Big-M)
    for j in J:
        for jp in J:
            if j < jp:
                for i in OJ[j]:
                    for ip in OJ[jp]:
                        common_workers = set(operations_workers[j, i]) & \
                                         set(operations_workers[jp, ip])
                        for k in common_workers:
                            model.addConstr(
                                s[jp, ip] >= s[j, i] +
                                operations_times[j, i, k]
                                - largeM * (3 - y[j, i, jp, ip]
                                            - x[j, i, k]
                                            - x[jp, ip, k]),
                                name=f"no_overlap1({j},{i},{jp},{ip},{k})"
                            )

                            model.addConstr(
                                s[j, i] >= s[jp, ip] +
                                operations_times[jp, ip, k]
                                - largeM * (2 + y[j, i, jp, ip]
                                            - x[j, i, k]
                                            - x[jp, ip, k]),
                                name=f"no_overlap2({j},{i},{jp},{ip},{k})"
                            )

    # cmax
    for j in J:
        last_i = max(OJ[j])
        model.addConstr(
            cmax >= s[j, last_i] +
            quicksum(operations_times[j, last_i, k] * x[j, last_i, k]
                     for k in operations_workers[j, last_i]),
            name=f"cmax({j})"
        )

    # --------------------
    # Solve
    # --------------------
    model.Params.TimeLimit = time_limits
    model.Params.OutputFlag = 0

    t0 = time.time()
    model.optimize()
    t1 = time.time()

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        return model.ObjVal * 1.6, t1 - t0
    else:
        return float("inf"), t1 - t0


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """
        Print intermediate solutions.
    """

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """
            Called at each new solution.
        """
        # print('Solution %i, time = %f s, objective = %i' %
        #       (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


if __name__ == '__main__':
    solve_instances(config=configs)