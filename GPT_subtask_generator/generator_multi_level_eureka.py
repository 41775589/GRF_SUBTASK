import copy
import logging
import sys
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from pathlib import Path
import shutil
import time
import re
from utils.misc import *
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task, create_train_cfg
from utils.extract_task_code import *
from copy import deepcopy
import collections
import yaml
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
MEDoE_DIR = os.path.abspath(os.path.join(current_dir, "../"))

SRC_DIR = os.path.join(MEDoE_DIR, "doe_epymarl-main/src")
sys.path.append(SRC_DIR)

from run import *


OpenAI.api_base = "https://api.ohmygpt.com"
client = OpenAI(api_key="key")

ROOT_DIR = os.getcwd()
parent_dir = os.path.dirname(ROOT_DIR)
GFOOTBALL_DIR = os.path.join(MEDoE_DIR, "gfootball")
CONFIG_ENVS_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/config/envs')
CONFIG_ALGS_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/config/algs')
MAP_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/maps')
REWARD_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/rewards')
prompt_dir = f'{ROOT_DIR}/utils/prompts'
logging.basicConfig(level=logging.INFO)

Time = datetime.datetime.now()
Time = Time.strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR = f"outputs/{Time}"
MAP_DIR = f"{MAP_DIR}/{Time}"
REWARD_DIR = f"{REWARD_DIR}/{Time}"
SACRED_DIR = os.path.join(MEDoE_DIR, "doe_epymarl-main/results/sacred")
TENSORBOARD_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/results/tb_logs')
# GRF_SCENARIO_DIR = f"{MEDoE_DIR}/scenarios/{Time}"

# 创建目标文件夹（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(REWARD_DIR, exist_ok=True)
# os.makedirs(GRF_SCENARIO_DIR, exist_ok=True)



def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def merge_doe_cls(groups, n_agents, role_list, doe_path, merge_doe_name, max_reward_code_path_for_each_group):
    # 初始化合并后的分类器
    merged_classifier = None
    merge_id = 0

    # 遍历每个组，加载对应的 DoE 分类器
    for group in groups:
        # 构建文件路径, 0_classifier.pt
        group_id = group["group_number"] - 1
        max_reward_code_path = max_reward_code_path_for_each_group[f"group{group_id}"].replace("reward", "cls").replace(
            ".py", ".pt")
        classifier_path = f"{doe_path}/{max_reward_code_path}"

        # 加载分类器
        classifier_i = torch.load(classifier_path)

        # 创建初始化一个 merged cls，因为load可以直接加载原来的类的所有属性，我们只需要扩展classifier_i的mlps尺寸，更新 self.n_agents即可
        # 避免重新指定各种网络参数
        if merged_classifier is None:
            # merged_classifier = doe_classifier_config_loader(n_agents, merge_cfg, doe_path, load_mode='merge')
            merged_classifier = copy.deepcopy(classifier_i)
            merged_classifier["n_agents"] = n_agents
            merged_classifier["role_list"] = role_list

            # for key in vars(merged_classifier).keys():
            #     print(key)
            # 扩展 lr 和 mlps 的数量
            merged_classifier["learning_rates"] = [merged_classifier["learning_rates"][0]] * n_agents
            merged_classifier["mlps"] = [merged_classifier["mlps"][0]] * n_agents

        # # 确保当前分类器的 mlps 列表长度与合并后的代理数量一致
        # assert classifier.n_agents == len(classifier1.mlps) + len(classifier2.mlps)

        # 合并历史分类器的参数到当前分类器中
        for doe_i in classifier_i["mlps"]:
            merged_classifier["mlps"][merge_id].load_state_dict(doe_i.state_dict())
            merge_id += 1

    assert merge_id == n_agents
    # 保存合并后的分类器
    print(merged_classifier)
    torch.save(merged_classifier, f'{doe_path}/{merge_doe_name}.pt')


# # 处理长文本，确保生成的 YAML 不包含复杂键
# def normalize_keys(data):
#     if isinstance(data, dict):
#         return {str(k): normalize_keys(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [normalize_keys(i) for i in data]
#     else:
#         return data

def train_merge_team(groups, is_doe, layer, decompose_id, group_id, iter_id, sample_id, buffer_dir, max_reward_code_path_for_each_group, Time, task_env, suffix, rl_runs, alg_cfg):
    team_structure = {
        "total_members": 0,
        "num_subteams": len(groups),
        "task_assignments": {}
    }

    # 记录当前的队员 ID
    current_id = 0

    # 遍历每个 group，将信息合并
    for group in groups:
        group_id = group["group_number"] - 1
        num_agents = group["number_of_agents"]

        # 更新总成员数量
        team_structure["total_members"] += num_agents

        # 为每个任务分配队员 ID
        task_assignments = {
            "task": f"goal_{group_id}",
            "member_ids": list(range(current_id, current_id + num_agents))
        }

        # 更新当前 ID
        current_id += num_agents

        # 将任务分配信息添加到队伍结构中
        team_structure["task_assignments"][f"group_{group_id}"] = task_assignments

    # {
    #     "total_members": 8,
    #     "num_subteams": 2,
    #     "task_assignments": {
    #         "group_1": {
    #             "task": "攻防训练",
    #             "member_ids": [0, 1, 2, 3, 4]
    #         },
    #         "group_2": {
    #             "task": "进攻训练",
    #             "member_ids": [5, 6, 7]
    #         },
    #     }
    # }

    role_list = []
    # 初始化任务 ID 计数器
    task_id_counter = 0

    # 遍历每个子团队，提取任务信息
    for group_key, group_info in team_structure["task_assignments"].items():
        member_ids = group_info["member_ids"]

        # 为每个成员添加对应的任务 ID
        role_list.extend([task_id_counter] * len(member_ids))

        # 任务 ID 计数器加 1
        task_id_counter += 1

    # role_list = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]，可用于指定merged doe的role ids
    # [attack attack defend]

    # 把团队角色信息转为role ids
    role_ids = {}
    for agent_id, role in enumerate(role_list):
        task_name = list(team_structure["task_assignments"].values())[role]["task"]  # 获取子团队任务名称
        if task_name not in role_ids:
            role_ids[task_name] = []
        role_ids[task_name].append(agent_id)
    # role_ids:
    #   "defence":
    #   - 0
    #   - 1
    #   - 2
    #   "attack":
    #   - 3
    #   - 4

    """
    To LZH:
    这里需要考虑加相对路径，修改 template file path 的位置，以及template config name 可以换成ia2c，作为基础参数模版，可以用于训练非doe的
    """

    # 读取 ia2c_ns.yaml 作为模板,也可以用ia2c
    template_config_name = alg_cfg
    template_file_path = f'{SRC_DIR}/config/algs/{template_config_name}.yaml'
    with open(template_file_path, 'r', encoding='utf-8') as template_file:
        template_data = yaml.safe_load(template_file)

    # 修改模板数据以生成 doe_ia2c.yaml 格式
    template_data['mac'] = "doe_mac"  # 修改 mac
    template_data['target_update_interval_or_tau'] = 0.01  # 修改更新间隔
    template_data['learner'] = f"doe_{alg_cfg}_learner"  # 修改学习器
    template_data['entropy_coef'] = 0.01  # 修改熵系数
    template_data['use_rnn'] = True  # 使用 RNN
    template_data['critic_type'] = "ac_critic"  # 修改评论家类型
    template_data['name'] = f"doe_{alg_cfg}"  # 修改名称

    # 11111111111指定 merge 以后的 full team doe cls 存储名称
    merged_doe_name = f"doe_{template_config_name}_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}_merged"

    # In multi-layer: add current iter and sample and this layer and this decomposed id to save for the father training
    # save_current_layer_merged_doe_path = f"doe_{template_config_name}_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}"
    save_current_layer_merged_doe_path = f"cls_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}"

    # role_ids_normalized = normalize_keys(role_ids)

    # 添加 DoE 相关参数
    doe_params = {
        # In multi-layer: add current iter and sample and this layer and this decomposed id to save for the father training
        "layer_id": layer,
        "decomposition_id": decompose_id,
        "group_id": group_id,
        "iter_id": iter_id,
        "sample_id": sample_id,
        #################################################
        "use_doe": True,
        "time_stamp": Time,
        "doe_type": "mlp",
        "ent_coef": 1.0,
        "base_lr": 1.0,
        "base_ent": 1.0,
        "boost_lr_coef": 1.0,
        "boost_ent_coef": 1.0,
        "doe_classifier_cfg": {
            "doe_type": "mlp",
            "load_mode": "train",
            "save_classifier": True,  # 首次训练没有doe，不用save，不过这里已经是merge阶段，而且使用doe，那么肯定要true
            "load_doe_buffer_path": buffer_dir,
            "save_doe_name": f"{save_current_layer_merged_doe_path}.pt",
            "load_doe_name": f"{merged_doe_name}.pt",  # 用于训练 merge team 的 doe cls，直接 load
            "mlp": {
                "hidden_sizes": [128],
                "batch_size": 512,
                "test_fraction": 0.1,
                "learning_rate": 1e-2
            },
            "role_ids": role_ids
        }
    }

    template_data.update(doe_params)
    # 指定要写入的新的 YAML 文件路径, decompose_id 代表某一种分解方案/第N次分解尝试的名字
    merged_doe_config_name = f"doe_{template_config_name}_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}_merged"
    new_yaml_file_path = f'{MEDoE_DIR}/doe_epymarl-main/src/config/algs/{merged_doe_config_name}.yaml'

    # 将修改后的数据写入新的 YAML 文件
    with open(new_yaml_file_path, 'w', encoding='utf-8') as new_yaml_file:
        yaml.dump(template_data, new_yaml_file, allow_unicode=True)

    print(f"New DOE YAML File {merged_doe_config_name}")

    # merge doe cls，保存到cfg.merge_doe_name
    merge_cfg_doe_params = template_data["doe_classifier_cfg"]
    merge_doe_cls(groups, team_structure["total_members"], role_list, buffer_dir, merged_doe_name,
                  max_reward_code_path_for_each_group)

    """本来考虑merge buffer再用于train doe cls，现在通过修改run中的加载doe逻辑，直接在每次训练中save cls和merge cls，不用再对齐buffer数据维度"""
    # # 处理buffer合并，用于doe training
    # """这里相对路径要修改"""
    # from components.episode_buffer import ReplayBuffer

    # # 加载两个 buffer
    # # buffer_dir = 'GRF_SUBTASK/doe_epymarl-main/results/buffer/grf'
    # buffer1 = torch.load(buffer_dir+'/buffer1.pt')
    # buffer2 = torch.load(buffer_dir+'/buffer2.pt')

    # total_agents = team_structure['total_members']  # 总团队的 agent 数量
    # doe_buffer = ReplayBuffer(scheme=buffer1.scheme,
    #                         groups={**buffer1.groups, **buffer2.groups},
    #                         buffer_size=total_agents,
    #                         max_seq_length=buffer1.max_seq_length)

    # # 将 buffer1 的数据插入到新的 buffer 中
    # doe_buffer.insert_episode_batch(buffer1)

    # # 调整 buffer2 的 agent ID
    # adjusted_buffer2_data = {}
    # for key in buffer2.data.transition_data.keys():
    #     adjusted_buffer2_data[key] = buffer2.data.transition_data[key].clone()
    #     """这里需要调整所有的key id"""
    #     if key == "actions":  # 假设 actions 需要调整
    #         adjusted_buffer2_data[key] += buffer1.groups["team_1"]  # 将 agent ID 调整

    # # 将调整后的 buffer2 数据插入到新的 buffer 中
    # doe_buffer.update(adjusted_buffer2_data,
    #                 slice(doe_buffer.buffer_index, doe_buffer.buffer_index + buffer2.batch_size),
    #                 slice(0, buffer2.max_seq_length))

    # 开始 train full team 在原始任务上
    # 默认如果用doe了，那么就是完全都用doe调节训练过程参数；如果不用doe，那么就是作为对比baseline
    """ To LZH
    这个环境名字目前写死 gfootball， smac 时可以改"""

    if is_doe:
        if layer == "target":
            rl_logpath = f"{OUTPUT_DIR}/{task_env}{suffix}_full_training_doe.txt"
            print(f'--config={merged_doe_config_name}')
            print(f'--env-config={task_env}')
            with open(rl_logpath, 'w') as f:
                script_path = f'{SRC_DIR}/main.py'
                params = [
                    'python', '-u', script_path,
                    f'--config={merged_doe_config_name}',
                    f'--env-config={task_env}',
                ]
                full_process = subprocess.Popen(params, stdout=f, stderr=f)
                full_process.wait()
                rl_runs = []
            # block_until_training(rl_logpath, log_status=True, iter_num=iter, response_id=response_id)
        else:
            TIMEOUT = 30
            # Execute the python file with flags
            rl_filepath = f"{OUTPUT_DIR}/{task_env}{suffix}_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}.txt"
            with open(rl_filepath, 'w') as f:
                script_path = f'{SRC_DIR}/main.py'
                params = [
                    'python', '-u', script_path,
                    f'--config={merged_doe_config_name}',
                    f'--env-config={task_env}{suffix}_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}',
                ]
                full_process = subprocess.Popen(params)

                # 获取文件的初始修改时间
                while True:
                    initial_mtime = os.path.getmtime(rl_filepath)
                    initial_mtime = datetime.datetime.fromtimestamp(initial_mtime)  # 时间转为datetime格式
                    start_time = datetime.datetime.now()
                    delta_time = start_time - initial_mtime  # 时间差
                    delta_seconds = delta_time.total_seconds()  # 时间差转成秒
                    if delta_seconds > TIMEOUT:  # 如果文件更新时间大于30秒，重新启动程序
                        print(
                            f"Overtime：It seems that the training is stuck or finished, subprocess terminates")
                        full_process.kill()  # 终止子进程
                        break
                    # while process.poll() is None:  # 检查子进程是否还在运行
                    #     # 检查文件的最后修改时间
                    #     current_mtime = os.path.getmtime(rl_filepath)
                    #     # 如果文件超过了 1 分钟没有更新
                    #     if current_mtime == initial_mtime and (time.time() - start_time) > TIMEOUT:
                    #         print(f"Overtime：It seems that the training is stuck, subprocess terminates")
                    #         process.terminate()  # 终止子进程
                    #         break
                    # 等待一段时间后再检查
                    time.sleep(1)

                full_process.wait()
            # Modified the check of successful training
            # block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(full_process)

    else:
        rl_logpath = f"{OUTPUT_DIR}/{task_env}{suffix}_full_training.txt"
        with open(rl_logpath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={template_config_name}',
                f'--env-config={task_env}',
            ]
            full_process = subprocess.Popen(params, stdout=f, stderr=f)
            full_process.wait()
        # block_until_training(rl_logpath, log_status=True, iter_num=iter, response_id=response_id)

    full_rl_training_performance = []
    full_rl_training_performance.append(full_process)
    # 似乎也不用save一个performance，tensorboard会自动生成的，就是找起来麻烦，要考虑一下logger的file合并
    # save(full_rl_training_performance)

    print('Merged Full Training Has Finished')
    return rl_runs




def main(model, n_decomposition, n_reward, temperature, task_env, alg_cfg, use_doe, n_improve_iter):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    suffix = "_GPT"
    logging.info(f"Using LLM: {model}")

    # env_init = f'{ROOT_DIR}/env_code/__init__.py'
    env = f'{ROOT_DIR}/env_code/{task_env}/football_env.py'
    env_core = f'{ROOT_DIR}/env_code/{task_env}/football_env_core.py'
    # action_set= f'{ROOT_DIR}/env_code/{task_env}/football_action_set.py'
    observation_processor = f'{ROOT_DIR}/env_code/{task_env}/observation_processor.py'
    scenario_builder = f'{ROOT_DIR}/env_code/{task_env}/scenario_builder.py'
    reward_wrapper_example = f'{ROOT_DIR}/env_code/{task_env}/reward_wrapper_example.py'
    obs_o = f'{ROOT_DIR}/env_code/{task_env}/obs_o.py'
    obs_exp = f'{ROOT_DIR}/env_code/{task_env}/obs_exp.py'
    example_of_task_tree = f'{ROOT_DIR}/env_code/{task_env}/task_tree_example.py'
    example_of_task_tree_feedback = f'{ROOT_DIR}/env_code/{task_env}/task_tree_feedback_example.py'

    # config = f'{ROOT_DIR}/env_code/{task_env}/config.py'
    # wrappers = f'{ROOT_DIR}/env_code/{task_env}/wrappers.py'

    # env_init_code_string  = file_to_string(env_init)
    env_code_string = file_to_string(env)
    env_core_code_string = file_to_string(env_core)
    observation_processor_code_string = file_to_string(observation_processor)

    env_code = env_code_string + env_core_code_string + observation_processor_code_string

    scenario_builder_code_string = file_to_string(scenario_builder)
    # wrappers = file_to_string(wrappers)

    reward_wrapper_example = file_to_string(reward_wrapper_example)
    obs_o = file_to_string(obs_o)
    obs_exp = file_to_string(obs_exp)
    example_of_task_tree = file_to_string(example_of_task_tree)
    example_of_task_tree_feedback = file_to_string(example_of_task_tree_feedback)

    three_vs_one_with_keeper = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_3_vs_1_with_keeper.py'
    corner = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_corner.py'
    counterattack_easy = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_counterattack_easy.py'
    counterattack_hard = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_counterattack_hard.py'
    empty_goal = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_empty_goal.py'
    empty_goal_close = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_empty_goal_close.py'
    pass_and_shoot_with_keeper = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_pass_and_shoot_with_keeper.py'
    run_pass_and_shoot_with_keeper = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_run_pass_and_shoot_with_keeper.py'
    run_to_score = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_run_to_score.py'
    run_to_score_with_keeper = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_run_to_score_with_keeper.py'
    single_goal_versus_lazy = f'{ROOT_DIR}/scenario_examples/{task_env}/academy_single_goal_versus_lazy.py'
    five_vs_five = f'{ROOT_DIR}/scenario_examples/{task_env}/5_vs_5.py'

    three_vs_one_with_keeper_code_string = file_to_string(three_vs_one_with_keeper)
    corner_code_string = file_to_string(corner)
    counterattack_easy_code_string = file_to_string(counterattack_easy)
    counterattack_hard_code_string = file_to_string(counterattack_hard)
    empty_goal_code_string = file_to_string(empty_goal)
    empty_goal_close_code_string = file_to_string(empty_goal_close)
    pass_and_shoot_with_keeper_code_string = file_to_string(pass_and_shoot_with_keeper)
    run_pass_and_shoot_with_keeper_code_string = file_to_string(run_pass_and_shoot_with_keeper)
    run_to_score_code_string = file_to_string(run_to_score)
    run_to_score_with_keeper_code_string = file_to_string(run_to_score_with_keeper)
    single_goal_versus_lazy_code_string = file_to_string(single_goal_versus_lazy)
    five_vs_five_code_string = file_to_string(five_vs_five)

    # parent_dir = os.path.dirname(ROOT_DIR)
    # parent_dir = os.path.dirname(parent_dir)
    # output_file_scenario = f"{parent_dir}/scenarios/scenario_{suffix.lower()}.py"

    # Loading all text prompts
    initial_system_further_decomposition = file_to_string(f'{prompt_dir}/{task_env}/initial_system_further_decomposition.txt')
    initial_user_further_decomposition = file_to_string(f'{prompt_dir}/{task_env}/initial_user_further_decomposition.txt')
    initial_system_get_decomposition = file_to_string(f'{prompt_dir}/{task_env}/initial_system_get_decomposition.txt')
    initial_user_get_decomposition = file_to_string(f'{prompt_dir}/{task_env}/initial_user_get_decomposition.txt')
    initial_system_scenarios = file_to_string(f'{prompt_dir}/{task_env}/initial_system_scenarios.txt')
    initial_user_scenarios = file_to_string(f'{prompt_dir}/{task_env}/initial_user_scenarios.txt')
    initial_system_rewards = file_to_string(f'{prompt_dir}/{task_env}/initial_system_rewards.txt')
    initial_user_rewards = file_to_string(f'{prompt_dir}/{task_env}/initial_user_rewards.txt')
    reward_signature = file_to_string(f'{prompt_dir}/{task_env}/reward_signature')

    execution_error_feedback = file_to_string(f'{prompt_dir}/{task_env}/execution_error_feedback.txt')
    code_feedback = file_to_string(f'{prompt_dir}/{task_env}/code_feedback.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/{task_env}/policy_feedback.txt')

    example_scenarios = file_to_string(f'{prompt_dir}/{task_env}/example_scenarios.txt')
    example_scenarios = example_scenarios.format(
        three_vs_one_with_keeper_code_string=three_vs_one_with_keeper_code_string,
        corner_code_string=corner_code_string,
        counterattack_easy_code_string=counterattack_easy_code_string,
        counterattack_hard_code_string=counterattack_hard_code_string,
        empty_goal_code_string=empty_goal_code_string,
        empty_goal_close_code_string=empty_goal_close_code_string,
        pass_and_shoot_with_keeper_code_string=pass_and_shoot_with_keeper_code_string,
        run_pass_and_shoot_with_keeper_code_string=run_pass_and_shoot_with_keeper_code_string,
        run_to_score_code_string=run_to_score_code_string,
        run_to_score_with_keeper_code_string=run_to_score_with_keeper_code_string,
        single_goal_versus_lazy_code_string=single_goal_versus_lazy_code_string
    )

    example_rewards = file_to_string(f'{prompt_dir}/{task_env}/example_rewards.txt')
    example_rewards = example_rewards.format(
        reward_wrapper=reward_wrapper_example
    )
    example_of_o = file_to_string(f'{prompt_dir}/{task_env}/example_of_o.txt')
    example_of_o = example_of_o.format(obs_o=obs_o, obs_exp=obs_exp)

    code_output_tip_scenarios = file_to_string(f'{prompt_dir}/{task_env}/code_output_tip_scenarios.txt')
    code_output_tip_rewards = file_to_string(f'{prompt_dir}/{task_env}/code_output_tip_rewards.txt')
    rule_setting = file_to_string(f'{prompt_dir}/{task_env}/rule_setting.txt')

    main_task = "learn to play a 11 vs 11 football game"
    num_agents = 11
    num_groups = 2

    initial_system_get_decomposition = initial_system_get_decomposition.format(rule_setting=rule_setting)
    initial_user_get_decomposition = initial_user_get_decomposition.format(main_task=main_task, num_agents=num_agents,
                                                                           num_groups=num_groups)

    messages = [{"role": "system", "content": initial_system_get_decomposition},
                {"role": "user", "content": initial_user_get_decomposition}]


    # Get response
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    # chunk_size = sample if "gpt-3.5" in model else 4
    chunk_size = n_decomposition

    logging.info(f"Decompositions Generation: Generating {n_decomposition} samples with {model}")

    while True:
        if total_samples >= n_decomposition:
            break
        for attempt in range(1000):
            print("ATTEMPT:", attempt)
            try:
                response_cur = client.chat.completions.create(model=model,
                                                              messages=messages,
                                                              temperature=temperature,
                                                              n=chunk_size)
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        responses.extend(response_cur.choices)
        print("RESPONSES:", responses)
        prompt_tokens = response_cur.usage.prompt_tokens
        total_completion_token += response_cur.usage.completion_tokens
        total_token += response_cur.usage.total_tokens

    # n_dec 代表分解多少个tree，本py用于单层深度分解
    if n_decomposition == 1:
        logging.info(f"Decompositions Generation: GPT Output:\n " + responses[0].message.content + "\n")

    # Logging Token Information
    logging.info(
        f"Decompositions Generation: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")


    """修改plan: response 0 和 1 的循环，添加一个check，如果是0，就调用ia2c训练并保存buffer/ckpt/yaml信息
        如果是1，并且 is doe true，那么利用子团队yaml信息创建doe的yaml，调用进行训练
        问题是，这样可能需要early stop或者某种metric，不知道是reward不好还是doe的不好（需要一种缺保doe是expert的判断条件）"""




    for response_id in range(n_decomposition):

        def get_task_branch(task_tree, task):
            """
            递归回溯，从 task 开始向上找到完整的任务链（branch）。
            """
            branch = []
            current_task = task

            while current_task:
                branch.append(current_task)
                father_group = current_task.get("father_group_number")

                if father_group is None:
                    break  # 找到根任务，停止回溯

                # 在所有层级查找 father_group 对应的任务
                found = False
                for layer in sorted(task_tree.keys(), reverse=True):  # 从高层到低层查找
                    for upper_task in task_tree[layer]:
                        if upper_task["group_number"] == father_group:
                            current_task = upper_task
                            found = True
                            break
                    if found:
                        break  # 退出外层循环

                if not found:
                    current_task = None  # 父任务不存在，终止回溯

            branch.reverse()  # 确保 branch 顺序从 root 到 leaf
            return branch

        def process_task_tree(task_tree, max_group_number):
            """
            处理任务树中 layer 最高的任务，逐层向上提取完整 branch，
            并调用 GPT 判断是否需要细分，若是则修改该任务并添加子任务。
            """
            # 找到当前最高层的任务
            max_layer = max(task_tree.keys())
            current_layer_tasks = task_tree[max_layer]

            # 逐个任务处理
            for task in current_layer_tasks:
                # 获取完整任务链（branch）
                branch = get_task_branch(task_tree, task)

                # 生成 GPT 任务描述
                cur_initial_system_further_dec = initial_system_further_decomposition
                cur_initial_user_further_dec = initial_user_further_decomposition.format(
                    current_branch_summary=branch
                )

                cur_messages_t = copy.deepcopy(messages)
                cur_messages_t.append({"role": "system", "content": cur_initial_system_further_dec})
                cur_messages_t.append({"role": "user", "content": cur_initial_user_further_dec})

                # 询问 GPT 是否需要细分
                gpt_response = client.chat.completions.create(
                    model=model,
                    messages=cur_messages_t,
                    temperature=temperature,
                    n=1
                )

                gpt_content = gpt_response.choices[0].message.content
                print(f"GPT Response for Task {task['group_number']} (Layer {max_layer}):\n{gpt_content}")

                # Extract GPT response information
                need_decomposition_match = re.search(r"\*\*Need further decomposition or not:\*\*\s*(.+)", gpt_content)
                new_training_goal_match = re.search(r"\*\*New Training goal:\*\*\s*(.+)", gpt_content)
                group_match = re.findall(
                    r"\*\*Group (\d+):\*\*\n\*\*Number of agents:\*\* (\d+)\n\*\*Training goal:\*\* (.+)", gpt_content)

                if not need_decomposition_match:
                    logging.warning(f"Failed to parse GPT response for Task {task['group_number']}!")
                    continue

                # Determine whether further decomposition is needed
                need_further_decomposition = need_decomposition_match.group(1).strip().lower() == "yes"

                # Skip if no further decomposition is needed
                if not need_further_decomposition:
                    continue

                # Modify the original task description if GPT provided a new one
                if new_training_goal_match:
                    task["training_goal"] = new_training_goal_match.group(1).strip()
                    # 更新任务树中的当前任务的训练目标
                    for layer_tasks in task_tree.values():
                        for task_in_tree in layer_tasks:
                            if task_in_tree["group_number"] == task["group_number"]:
                                task_in_tree["training_goal"] = task["training_goal"]
                                break

                # Parse and add child tasks
                if group_match:
                    child_tasks = []
                    for match in group_match:
                        new_group_number = max_group_number + 1
                        group_info = {
                            "group_number": new_group_number,
                            "number_of_agents": int(match[1]),
                            "training_goal": match[2].strip(),
                            "layer": max_layer + 1,
                            "father_group_number": task["group_number"]
                        }
                        child_tasks.append(group_info)
                        max_group_number = new_group_number

                    # Add child tasks to the task tree
                    if child_tasks:
                        task_tree.setdefault(max_layer + 1, []).extend(child_tasks)

            return task_tree, max_group_number

        def get_child_tasks(task_tree, layer, group_number):
            """
            获取 task_tree 中 layer+1 层中 father_group_number 为 group_number 的子任务列表。
            """
            next_layer = layer + 1
            if next_layer not in task_tree:
                return []  # 如果下一层不存在，返回空列表

            # 筛选 father_group_number 等于当前 group_number 的任务
            child_tasks = [
                task for task in task_tree[next_layer]
                if task["father_group_number"] == group_number
            ]
            return child_tasks

        # 主循环，处理所有初始分解
        print(f"Processing decomposition {response_id}...")
        # 初始化任务树
        task_tree = {}
        max_group_number = 0

        # 当前 response 的内容
        response_cur = responses[response_id].message.content

        # 正则表达式匹配分解信息
        pattern = r"\*\*Group (\d+):\*\*\n\*\*Number of agents:\*\* (\d+)\n\*\*Training goal:\*\* (.+)"
        matches = re.findall(pattern, response_cur)

        # 当前层的任务列表
        layer_info = []
        for match in matches:
            new_group_number = max_group_number + 1
            group_info = {
                "group_number": new_group_number,
                "number_of_agents": int(match[1]),
                "training_goal": match[2].strip(),
                "layer": 0,
                "father_group_number": None
            }
            layer_info.append(group_info)
            max_group_number = new_group_number

        task_tree[0] = layer_info

        while True:
            prev_group_count = max_group_number
            task_tree, max_group_number = process_task_tree(task_tree, max_group_number)


            # 如果任务数量没有变化，说明所有任务都已分解完毕，终止循环
            if prev_group_count == max_group_number:
                break



        # 打印任务树
        import pprint
        pprint.pprint({"Final Task Tree": task_tree})

        response_cur = responses[response_id].message.content
        # responses是 len=2 的list，每个都是dict

        # 这里的命名文件等待zihao更新，用于创建env和reward
        # response id 代表分解第几种分解方案，samples；layer0代表只分解一层
        with open(f"{OUTPUT_DIR}/decomposition{response_id}.py", 'w') as file:
            file.writelines(str(task_tree) + '\n')

#####################################################################################################################################

        max_layer = max(task_tree.keys()) if task_tree else 0
        max_reward_code_path_for_each_group = {}

        # 从最大层向上遍历
        for layer in range(max_layer, -1, -1):
            print(f"Processing Layer {layer}...")
            tasks = task_tree[layer]

            # 分别训练两个子团队任务
            # if use_doe, 每个子团队任务训练结束后的buffer要保存，用于训练 doe classifier

            # 用于存储每个group最终选择的最佳奖励函数的路径，训练上层doe时读取对应路径的buffer使用

            for task in tasks:
                print(f"Processing Task {task['group_number']} from Layer {layer}")
                group_id = task["group_number"] - 1
                # Scenario generation
                logging.info(
                    f"Scenarios Generation: Generating 1 sample for Decomposition {response_id} Layer{layer} Group{group_id} with {model}")

                cur_initial_system_scenarios = initial_system_scenarios.format(
                    main_task_scenario=five_vs_five_code_string) + code_output_tip_scenarios + example_scenarios
                cur_initial_user_scenarios = initial_user_scenarios.format(training_goal=task['training_goal'],
                                                                           number_of_agents=task['number_of_agents'],
                                                                           scenario_builder_code_string=scenario_builder_code_string)
                current_tree = json.dumps(task_tree, indent=4)
                cur_messages_s = copy.deepcopy(messages)
                cur_messages_s.append({"role": "assistant", "content": f"Current entire task tree is:\n{current_tree}"})
                cur_messages_s.append({"role": "system", "content": cur_initial_system_scenarios})
                cur_messages_s.append({"role": "user", "content": cur_initial_user_scenarios})

                response_scenario_cur = client.chat.completions.create(model=model,
                                                                       messages=cur_messages_s,
                                                                       temperature=temperature,
                                                                       n=1)
                reply_scenario = response_scenario_cur.choices[0].message.content

                # 提取prompt和env代码，生成训练任务scenario

                # # Regex patterns to extract python code enclosed in GPT response
                # patterns = [
                #     r'```python(.*?)```',
                #     r'```(.*?)```',
                #     r'"""(.*?)"""',
                #     r'""(.*?)""',
                #     r'"(.*?)"',
                # ]
                # for pattern in patterns:
                #     scenario_code_string = re.search(pattern, reply_scenario, re.DOTALL)
                #     if scenario_code_string is not None:
                #         scenario_code_string = scenario_code_string.group(1).strip()
                #         break
                # scenario_code_string = reply_scenario if not scenario_code_string else scenario_code_string

                # print("Scenario Code String 1:", scenario_code_string)
                def extract_code_scenario(reply_scenario):
                    # Regex patterns to extract python code enclosed in GPT response
                    patterns = [
                        r'```python(.*?)```',
                        r'```(.*?)```',
                        r'"""(.*?)"""',
                        r'""(.*?)""',
                        r'"(.*?)"',
                    ]
                    for pattern in patterns:
                        scenario_code_string = re.search(pattern, reply_scenario, re.DOTALL)
                        if scenario_code_string is not None:
                            return scenario_code_string.group(1).strip()
                    return reply_scenario

                # # Remove unnecessary imports
                # lines = scenario_code_string.split("\n")
                # for i, line in enumerate(lines):
                #     if line.strip().startswith("def "):
                #         scenario_code_string = "\n".join(lines[i:])

                # Retry until the scenario has two goal keepers
                while True:
                    reply_scenario = response_scenario_cur.choices[0].message.content
                    scenario_code_string = extract_code_scenario(reply_scenario)

                    # Remove unnecessary imports
                    lines = scenario_code_string.split("\n")
                    for i, line in enumerate(lines):
                        if line.strip().startswith("def "):
                            scenario_code_string = "\n".join(lines[i:])

                    # Check for at least two occurrences of e_PlayerRole_GK
                    if scenario_code_string.count("e_PlayerRole_GK") == 2 and "e_PlayerRole." not in scenario_code_string:
                        break  # Satisfied condition, proceed to save the code
                    else:
                        print(f"Wrong Scenario Code with only one goal keeper or fully qualified names. Retrying...")
                        # Re-generate response
                        cur_messages_s.append({"role": "assistant",
                                               "content": "Regenerate the scenario, ensuring that:"
                                                          "1. Each team has **exactly one goalkeeper ('e_PlayerRole_GK')**."
                                                          "2. Do **NOT** use fully qualified names (FQN) like 'e_PlayerRole.e_PlayerRole_GK'. Instead, use **direct values** (e.g., 'e_PlayerRole_GK', 'e_PlayerRole_CF')."})
                        response_scenario_cur = client.chat.completions.create(model=model,
                                                                               messages=cur_messages_s,
                                                                               temperature=temperature,
                                                                               n=1)

                print("Scenario Code String 2:", scenario_code_string)

                # Save the new environment code when the output contains valid code string!
                with open(f"{MAP_DIR}/scenario_layer{layer}_decomposition{response_id}_subtask{group_id}.py", 'w') as file:
                    file.writelines("from . import *" + '\n')
                    file.writelines(scenario_code_string + '\n')

                with open(f"{OUTPUT_DIR}/scenario_layer{layer}_decomposition{response_id}_subtask{group_id}.py", 'w') as file:
                    file.writelines("from . import *" + '\n')
                    file.writelines(scenario_code_string + '\n')

                # Save the scenario in the GRF Env
                with open(f"{GFOOTBALL_DIR}/scenarios/scenario_layer{layer}_decomposition{response_id}_subtask{group_id}.py",
                          'w') as file:
                    file.writelines("from . import *" + '\n')
                    file.writelines(scenario_code_string + '\n')

                # 生成子任务的环境代码保存到py文件

                # Reward generation and improving:
                logging.info(
                    f"Rewards Generation: Generating {n_reward} samples for Decomposition {response_id} Layer{layer} Group{group_id} with {model}")

                curr_code_output_tip_rewards = code_output_tip_rewards.format(
                    number_of_agents=task['number_of_agents'],
                    example_of_o=example_of_o, reward_signature=reward_signature)
                cur_initial_system_rewards = initial_system_rewards + example_rewards + curr_code_output_tip_rewards
                cur_initial_user_rewards = initial_user_rewards.format(training_goal=task['training_goal'],
                                                                       number_of_agents=task['number_of_agents'],
                                                                       env_code=env_code, )

                cur_messages_r = copy.deepcopy(messages)
                cur_messages_r.append({"role": "assistant", "content": f"Current entire task tree is:\n{current_tree}"})
                # cur_messages_r.append({"role": "assistant", "content": reply_scenario})
                cur_messages_r.append({"role": "system", "content": cur_initial_system_rewards})
                cur_messages_r.append({"role": "user", "content": cur_initial_user_rewards})

                DUMMY_FAILURE = -10000.
                max_scores = []
                max_successes_reward_correlation = []
                execute_rates = []
                best_code_paths = []
                max_score_overall = DUMMY_FAILURE
                max_reward_code_path = None
                max_reward_code_path_for_each_group[f'group{group_id}'] = None

                # 尝试几次 reward 生成 batch，默认2
                for iter in range(n_improve_iter):
                    total_samples_r = 0
                    responses_r = []
                    chunk_size_r = n_reward

                    # n reward 为 1

                    while True:
                        if total_samples_r >= n_reward:
                            break
                        for attempt in range(1000):
                            print("ATTEMPT:", attempt)
                            try:
                                reply_rewards_cur = client.chat.completions.create(model=model,
                                                                                   messages=cur_messages_r,
                                                                                   temperature=temperature,
                                                                                   n=chunk_size_r)
                                total_samples_r += chunk_size
                                break
                            except Exception as e:
                                if attempt >= 10:
                                    chunk_size = max(int(chunk_size / 2), 1)
                                    print("Current Chunk Size", chunk_size)
                                logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                                time.sleep(1)
                        if reply_rewards_cur is None:
                            logging.info("Code terminated due to too many failed attempts!")
                            exit()

                        responses_r.extend(reply_rewards_cur.choices)
                    # responses r是一个list，用于存储根据message r询问LLM得到的reward，这里只cue 1次

                    ####################################
                    code_runs = []
                    rl_runs = []
                    #####################################

                    for response_r_id in range(n_reward):
                        reply_reward = responses_r[response_r_id].message.content
                        print("REPLY REWARD: ", reply_reward)
                        print("REWARD TOKEN:", reply_rewards_cur.usage.prompt_tokens)
                        # Regex patterns to extract python code enclosed in GPT response
                        patterns = [
                            r'```python(.*?)```',
                            r'```(.*?)```',
                            r'"""(.*?)"""',
                            r'""(.*?)""',
                            r'"(.*?)"',
                        ]
                        for pattern in patterns:
                            reward_code_string = re.search(pattern, reply_reward, re.DOTALL)
                            if reward_code_string is not None:
                                reward_code_string = reward_code_string.group(1).strip()
                                break
                        reward_code_string = reply_reward if not reward_code_string else reward_code_string

                        print("Reward Code String 1:", reward_code_string)

                        # Remove unnecessary imports
                        lines = reward_code_string.split("\n")
                        for i, line in enumerate(lines):
                            if line.strip().startswith("class "):
                                reward_code_string = "\n".join(lines[i:])

                        print("Reward Code String 2:", reward_code_string)

                        code_runs.append(reward_code_string)

                        # response id是分解几层，response r id是sample的reward function个数
                        with open(
                                f"{REWARD_DIR}/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.py",
                                'w') as file:
                            file.writelines("import gym" + '\n')
                            file.writelines("import numpy as np" + '\n')
                            file.writelines(reward_code_string + '\n')

                        with open(
                                f"{OUTPUT_DIR}/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.py",
                                'w') as file:
                            file.writelines("import gym" + '\n')
                            file.writelines("import numpy as np" + '\n')
                            file.writelines(reward_code_string + '\n')

                        # Save the reward function in the GRF Env
                        with open(
                                f"{GFOOTBALL_DIR}/rewards/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.py",
                                'w') as file:
                            file.writelines("import gym" + '\n')
                            file.writelines("import numpy as np" + '\n')
                            file.writelines(reward_code_string + '\n')


                        # 到此为止是生成了 for group -> for n_prove 几个reward -> for reward id 第几个reward的train feedback

                        ####################################################################################################################################################################################################
                        # # Find the freest GPU to run GPU-accelerated RL
                        # set_freest_gpu()

                        # """
                        # 这里alg_cfg需要根据分解的子任务，创建对应的doe_ia2c，也就是 doe_classifer_cfg/
                        #     # 2s3z/3m
                        #     role_ids:
                        #         defence:  # classifier.role_list=[0,1,1,0,0]
                        #             - 0 # agent id
                        #         attack:
                        #             - 2
                        #             - 1
                        # 在原始的doe代码中（目前版本），cfg文件表示的是两个子团队合并到一起时的任务分配列表
                        # 即将defence和attack两个子团队合并到一起进行训练时的config设定

                        # 而在每个子团队训练时，需要调用对应的子团队cfg，因此需要在创建子任务后生成各自的yaml文件
                        # 比如one level分解为group 1和group2，就需要两个不同的cfg
                        # 分别是 role_ids: defence: -0 和 role_ids: attack: -1

                        # 当然如果group 1 & group 2已经是最小的子任务的话，那么就不要调用doe_ia2c，
                        # 而是直接调用ia2c进行训练，并且在训练结束后存储各自的buffer.pt
                        # ref src/run.py Line 150
                        # 这个buffer pt会用于下次merge这两个子团队时train各自的doe classifier
                        # 这部分有待一起讨论
                        # """

                        # 在这里控制是保存buffer还是load buffer，修改 src/main 中 run 的逻辑
                        # 在单层分解中，为了简化过程，我们设定默认子任务训练都保存buffer
                        # 在多层分解中，需要额外考虑save/load逻辑
                        TIMEOUT = 30
                        #如果是最底层，不用doe
                        if layer == max_layer:
                            # Create Task YAML files
                            create_task(CONFIG_ENVS_DIR, task_env, layer, response_id, response_r_id, task['number_of_agents'],
                                        task['group_number'] - 1, iter, suffix)
                            create_train_cfg(CONFIG_ALGS_DIR, Time, alg_cfg, layer, response_id, response_r_id,
                                             task['number_of_agents'], task['group_number'] - 1, iter)
                            # Execute the python file with flags
                            rl_filepath = f"{OUTPUT_DIR}/{task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.txt"
                            with open(rl_filepath, 'w') as f:
                                script_path = f'{SRC_DIR}/main.py'
                                params = [
                                    'python', '-u', script_path,
                                    f'--config={alg_cfg}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}',
                                    f'--env-config={task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}',
                                ]
                                process = subprocess.Popen(params, stdout=f, stderr=f)

                                # 获取文件的初始修改时间
                                while True:
                                    initial_mtime = os.path.getmtime(rl_filepath)
                                    initial_mtime = datetime.datetime.fromtimestamp(initial_mtime)  # 时间转为datetime格式
                                    start_time = datetime.datetime.now()
                                    delta_time = start_time - initial_mtime  # 时间差
                                    delta_seconds = delta_time.total_seconds()  # 时间差转成秒
                                    if delta_seconds > TIMEOUT:  # 如果文件更新时间大于30秒，重新启动程序
                                        print(
                                            f"Overtime：It seems that the training is stuck or finished, subprocess terminates")
                                        process.kill()  # 终止子进程
                                        break
                                    # while process.poll() is None:  # 检查子进程是否还在运行
                                    #     # 检查文件的最后修改时间
                                    #     current_mtime = os.path.getmtime(rl_filepath)
                                    #     # 如果文件超过了 1 分钟没有更新
                                    #     if current_mtime == initial_mtime and (time.time() - start_time) > TIMEOUT:
                                    #         print(f"Overtime：It seems that the training is stuck, subprocess terminates")
                                    #         process.terminate()  # 终止子进程
                                    #         break
                                    # 等待一段时间后再检查
                                    time.sleep(1)

                                process.wait()
                            # Modified the check of successful training
                            # block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                            rl_runs.append(process)
                        else:
                            ###############Use doe for training##############
                            use_doe = True
                            create_task(CONFIG_ENVS_DIR, task_env, layer, response_id, response_r_id,
                                        task['number_of_agents'],
                                        task['group_number'] - 1, iter, suffix)
                            child_tasks = get_child_tasks(task_tree, layer, group_id + 1)
                            rl_runs = train_merge_team(child_tasks, use_doe, layer=layer, decompose_id=response_id,
                                             group_id = group_id, iter_id = iter, sample_id = response_r_id,
                                             buffer_dir=f'{MEDoE_DIR}/doe_epymarl-main/results/buffers/gfootball/{Time}',
                                             max_reward_code_path_for_each_group=max_reward_code_path_for_each_group,
                                             Time=Time, task_env=task_env, suffix=suffix, rl_runs = rl_runs, alg_cfg=alg_cfg)


                    # 完成了reward次数的RL training，收集了所有的traj

                    # Gather RL training results and construct reward reflection
                    code_feedbacks = []
                    contents = []
                    # May add other metrics
                    score_reward_mean = []
                    code_paths = []

                    # print("RRRRRRRRR",rl_runs)

                    exec_success = False
                    for response_r_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
                        rl_run.communicate()
                        rl_filepath = f"{OUTPUT_DIR}/{task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.txt"
                        result_path = f'{SACRED_DIR}/{Time}/scenario_layer{layer}_decomposition{response_id}_subtask{group_id}/scoring, reward_layer0_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}/1'
                        code_paths.append(
                            f"reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}.py")
                        try:
                            with open(rl_filepath, 'r') as f:
                                stdout_str = f.read()
                        except:
                            # content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                            content = execution_error_feedback.format(
                                traceback_msg="Code Run cannot be executed because reward class is wrongly formatted! Please re-write an entirely new reward function!")
                            content += code_output_tip_rewards.format(number_of_agents=task['number_of_agents'],
                                                                      example_of_o=example_of_o,
                                                                      reward_signature=reward_signature)
                            contents.append(content)
                            score_reward_mean.append(DUMMY_FAILURE)
                            continue

                        content = ''
                        traceback_msg = filter_traceback(stdout_str)
                        done_string = 'absl Dump "episode_done": count limit reached / disabled'
                        num_done_string = stdout_str.count(done_string)

                        if traceback_msg == '':
                            if num_done_string < 6:
                                print("Wrong scenario!")
                                score_reward_mean.append(DUMMY_FAILURE)
                                content += execution_error_feedback.format(
                                    traceback_msg="Wrong Scenario without Goalkeeper")
                            else:
                                print("No errors in the reward function")
                                # If RL execution has no error, provide policy statistics feedback
                                exec_success = True

                                # tensorboard_logdir = f"{TENSORBOARD_DIR}/{Time}/layer0_decomposition{response_id}_subtask{group_id}_iter{iter}_sample{response_r_id}"
                                # tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)

                                content += policy_feedback

                                with open(f"{result_path}/metrics.json", "r") as file:
                                    data = json.load(file)

                                corresponding_episodes = data["episode"]['values']
                                content += f"corresponding episodes: {corresponding_episodes[::10]} \n"

                                component_values = {
                                    key: value['values'] for key, value in data.items() if key.startswith('component')
                                }
                                for key, value in component_values.items():
                                    value = value[::10]
                                    content += f"{key}: {value} \n"

                                score_reward_mean_values = data["score_reward_mean"]['values']
                                # 获取最后10个值
                                last_10_values = score_reward_mean_values[-10:]
                                # 计算平均数
                                average = sum(last_10_values) / len(last_10_values)
                                score_reward_mean.append(average)

                                content += f"score_reward_mean: {score_reward_mean_values[::10]} \n"

                                final_reward_mean_values = data["final_reward_mean"]['values']
                                content += f"final_reward_mean: {final_reward_mean_values[::10]} \n"

                                # Here add metrics tracking
                                # content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"

                                code_feedbacks.append(code_feedback)
                                content += code_feedback
                                # The token is too long for the message with Traceback (error reward functions). So only good reward has this
                                content += code_output_tip_rewards.format(number_of_agents=task['number_of_agents'],
                                                                          example_of_o=example_of_o,
                                                                          reward_signature=reward_signature)
                        else:
                            print("Spotting errors in the reward function")
                            # Otherwise, provide execution traceback error feedback
                            score_reward_mean.append(DUMMY_FAILURE)
                            content += execution_error_feedback.format(traceback_msg=traceback_msg)

                        # The token is too long for the message with Traceback (error reward functions).
                        # content += code_output_tip_rewards.format(number_of_agents=task['number_of_agents'],
                        #                                           example_of_o=example_of_o,
                        #                                           reward_signature=reward_signature)
                        contents.append(content)

                    # Repeat the iteration if all code generation failed
                    if not exec_success and n_reward != 1:
                        execute_rates.append(0.)
                        max_scores.append(DUMMY_FAILURE)
                        best_code_paths.append(None)
                        logging.info(
                            "All code generation failed! Repeat this iteration from the current message checkpoint!")
                        continue

                    # Select the best code sample based on the success rate
                    # print("CCCCCCCCCCC",content)
                    best_sample_idx = np.argmax(np.array(score_reward_mean))
                    best_content = contents[best_sample_idx]

                    max_score = score_reward_mean[best_sample_idx]
                    # max_success_reward_correlation = reward_correlations[best_sample_idx]
                    execute_rate = np.sum(np.array(score_reward_mean) >= 0.) / n_reward

                    # Update the best Eureka Output
                    if max_score > max_score_overall:
                        max_score_overall = max_score
                        max_reward_code_path = code_paths[best_sample_idx]
                        max_reward_code_path_for_each_group[f'group{group_id}'] = max_reward_code_path

                    execute_rates.append(execute_rate)
                    max_scores.append(max_score)
                    best_code_paths.append(code_paths[best_sample_idx])

                    logging.info(f"Iteration {iter}: Max Score: {max_score}, Execute Rate: {execute_rate}")
                    logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
                    logging.info(f"Iteration {iter}: GPT Output Content:\n" + responses_r[
                        best_sample_idx].message.content + "\n")
                    logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

                    # Modify: check assistent contents
                    if len(cur_messages_r) == 5:
                        cur_messages_r += [
                            {"role": "assistant", "content": responses_r[best_sample_idx].message.content}]
                        cur_messages_r += [{"role": "user", "content": best_content}]
                    else:
                        assert len(cur_messages_r) == 7
                        cur_messages_r[-2] = {"role": "assistant",
                                              "content": responses_r[best_sample_idx].message.content}
                        cur_messages_r[-1] = {"role": "user", "content": best_content}

                if max_reward_code_path is None:
                    logging.info("All iterations of code generation failed, aborting...")
                    logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
                    exit()

        # 完成了 所有子任务 reward 生成，开始train merge team
        """
        上面第一阶段子任务训练 alg_cfg 需要用 ia2c/ia2c_ns，不能用 doe 干扰RL训练
        但是需要 save buffer 并得到 doe cls，然后在第二阶段 merge train 时 merge cls
        这种写法是每个子任务自己的性能提升自己的表现，先用着，以后的研究中再考虑与merge后技能的表现
        """

        """To LZH
        这里暂时采用的绝对路径 buffer_dir 其实就是一组分解plan中每个阶段的doe相关数据，
        比如分解方案一，分解一层两队，
        路径就是 results/buffer/grf/decomposition0+ layer0_group0_buffer.pt & layer0_group0_doe.pt & layer0_group1_doe.pt etc.
        需要调整对应的命名方式，上面只是举例，

        以及这里 decompose_id = 0 是按照第0个decompose方案的意思设计的，如果不对可以改，主要影响 yaml 文件命名
        """
        use_doe=True
        print("Start merging and training on the target task")
        rl_runs = train_merge_team(task_tree[0], use_doe, layer="target", decompose_id=response_id, group_id = "target", iter_id = "target",
                                   sample_id = "target", buffer_dir=f'{MEDoE_DIR}/doe_epymarl-main/results/buffers/{task_env}/{Time}',
                                   max_reward_code_path_for_each_group=max_reward_code_path_for_each_group, Time=Time,
                                   task_env = task_env, suffix = suffix, rl_runs = [], alg_cfg=alg_cfg)

    # 完成了所有方案 n decomposition plan 的任务生成，Execute the Main task using w/w. DOE:



if __name__ == "__main__":
    main(model="gpt-4-turbo", n_decomposition=1, n_reward=5, temperature=1, task_env="gfootball", alg_cfg="ia2c",
         use_doe=False, n_improve_iter=2)