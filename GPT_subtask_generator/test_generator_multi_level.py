import copy
import logging
import sys
import os
import datetime
from collections import defaultdict, OrderedDict

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
RESULTS_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/results')
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
    """
    Func:
    加载并合并所有groups的doe classifier为一个新的doe cls

    Params:
    groups: dict - 提供分组id
    n_agents: int - 合并后团队总 agents 数量
    role_list: list - 合并后任务分工 # [0, 0, 1, 1, 2]
    doe_path: dir - 存储路径  # f'GRF_SUBTASK/doe_epymarl-main/results/gfootball/Time/decomposition0/group0'
    merge_doe_name: dir - 存储合并后doe cls路径 # f"doe_'template_config_name'_layer'layer'_decomposition'decompose_id'_subtask{group_id}_iter{iter_id}_sample{sample_id}_merged"
    max_reward_code_path_for_each_group: dir - 替换py为pt，指定load上一层的doe cls

    Outputs:
    合并后的 doe cls file: f'{doe_path}/{merge_doe_name}.pt')
    """

    """
    Func:
    加载并合并所有groups的doe classifier为一个新的doe cls

    Params:
    groups: dict - 提供分组id
    n_agents: int - 合并后团队总 agents 数量
    role_list: list - 合并后任务分工 # [0, 0, 1, 1, 2]
    doe_path: dir - 存储路径  # f'GRF_SUBTASK/doe_epymarl-main/results/gfootball/Time/decomposition0/group0'
    merge_doe_name: dir - 存储合并后doe cls路径 # f"doe_'template_config_name'_layer'layer'_decomposition'decompose_id'_subtask{group_id}_iter{iter_id}_sample{sample_id}_merged"
    max_reward_code_path_for_each_group: dir - 替换py为pt，指定load上一层的doe cls

    Outputs:
    合并后的 doe cls file: f'{doe_path}/{merge_doe_name}.pt')
    """

    merged_classifier = None
    merge_id = 0

    for group in groups:
        group_id = group["group_number"] - 1

        # 用于设置cls的文件名
        child_group_dir = os.path.join(os.path.dirname(doe_path), f"group{group_id}")
        doe_cls_file_name = max_reward_code_path_for_each_group[f"group{group_id}"].replace("reward", "cls").replace(
            ".py", ".pt")
        classifier_path = os.path.join(child_group_dir, doe_cls_file_name)
        # classifier_path = f"{doe_path}/{max_reward_code_path}"
        # projects/GRF_SUBTASK/doe_epymarl-main/results/gfootball/0505_ia2c/decomposition0/group6/cls_layer2_decomposition0_subtask6_iter0_sample0.pt

        # 加载上一层训练好的分类器
        classifier_i = torch.load(classifier_path, weights_only=False)

        # 创建初始化一个 n agents merged cls，load cls 避免重新指定各种网络参数
        if merged_classifier is None:
            # merged_classifier = doe_classifier_config_loader(n_agents, merge_cfg, doe_path, load_mode='merge')
            merged_classifier = copy.deepcopy(classifier_i)
            merged_classifier["n_agents"] = n_agents
            merged_classifier["role_list"] = role_list

            # for key in vars(merged_classifier).keys():
            #     print(key)

            # 扩展 lr 和 mlps 的数量，创建 n_agents list mlps
            merged_classifier["learning_rates"] = [merged_classifier["learning_rates"][0]] * n_agents
            merged_classifier["mlps"] = [merged_classifier["mlps"][0]] * n_agents

        # 加载历史分类器的参数到当前分类器id中
        for doe_i in classifier_i["mlps"]:
            merged_classifier["mlps"][merge_id].load_state_dict(doe_i.state_dict())
            merge_id += 1

    assert merge_id == n_agents
    torch.save(merged_classifier, os.path.join(doe_path, f"{merge_doe_name}.pt"))

def split_actors_by_agent(merged_actor):
    """将 merged_actor 中的多个 OrderedDict 拆分为单 agent 的 OrderedDict"""
    new_merged_actor = []

    # 遍历每个子任务的模型（每个模型是一个 OrderedDict）
    for actor_model in merged_actor:
        # 用 defaultdict 创建一个存储 agent 信息的字典
        agents_dict = defaultdict(OrderedDict)

        # 遍历当前模型中的所有键值对
        for key, value in actor_model.items():
            # 假设 key 的格式是 agents.n.some_property
            if key.startswith('agents.') and '.' in key[7:]:  # 排除不符合 agents.n 格式的 key
                agent_prefix = key.split('.')[1]  # 提取 'n'，例如 '0'、'1' 等
                agents_dict[agent_prefix][key] = value  # 将键值对添加到对应的 agent 中
            else:
                print(f"Warning: Key '{key}' does not contain a valid agent prefix.")

        # 将每个 agent 的 OrderedDict 添加到新的结果列表中
        new_merged_actor.extend(agents_dict.values())

    return new_merged_actor


def merge_policy(groups, buffer_dir):
    """
    merge多个 group 的 actor/critic 模型，保存为列表形式，用于初始化medoe训练的策略。
    Params:
        groups: list of dicts, 每个包含 group_number, number_of_agents
        buffer_dir: str, 当前父task的路径
    """
    import torch
    import os

    # 存储所有的actor和critic状态list
    merged_actor = []
    merged_critic = []
    n_total_agents = 0

    for group in groups:
        group_id = group["group_number"] - 1
        group_path = os.path.join(os.path.dirname(buffer_dir), f"group{group_id}")
        n_total_agents += group["number_of_agents"]
        actor_path = os.path.join(group_path, "agent.th")
        critic_path = os.path.join(group_path, "critic.th")

        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            raise FileNotFoundError(f"Missing actor or critic model in group{group_id}")

        actor_model = torch.load(actor_path)
        critic_model = torch.load(critic_path)

        merged_actor.append(actor_model)

        print("Critic model:",critic_model)
        print("len of critic model",len(critic_model))

        for i in range(len(critic_model)):
            merged_critic.append(critic_model[i])

    merged_actor = split_actors_by_agent(merged_actor)
    assert len(merged_actor) == n_total_agents
    # os.makedirs(merged_policy_dir, exist_ok=True)
    actor_save_path = os.path.join(buffer_dir, "actor_init.th")
    critic_save_path = os.path.join(buffer_dir, "critic_init.th")

    torch.save(merged_actor, actor_save_path)
    torch.save(merged_critic, critic_save_path)

    print(f"Merged actor saved to {actor_save_path}")
    print(f"Merged critic saved to {critic_save_path}")

# def merge_policy(groups, buffer_dir):
#     """
#     merge多个 group 的 actor/critic 模型，平均策略参数
#     Params:
#         groups: list of dicts, 每个包含 group_number, number_of_agents
#         buffer_dir: str, 当前父task的路径
#     """
#     import torch
#     import os

#     # 存储所有的actor和critic状态list
#     actor_states = []
#     critic_states = []
#     n_total_agents = 0

#     for group in groups:
#         group_id = group["group_number"] - 1
#         group_path = os.path.join(os.path.dirname(buffer_dir), f"group{group_id}")
#         n_total_agents += group["number_of_agents"]
#         actor_path = os.path.join(group_path, "agent.th")
#         critic_path = os.path.join(group_path, "critic.th")


#         if not os.path.exists(actor_path) or not os.path.exists(critic_path):
#             raise FileNotFoundError(f"Missing actor or critic model in group{group_id}")

#         actor_model = torch.load(actor_path)
#         critic_model = torch.load(critic_path)

#         actor_states.append(actor_model)
#         critic_states.append(critic_model)

#     # assert len(merged_actor) == n_total_agents
#     # os.makedirs(merged_policy_dir, exist_ok=True)

#     def average_state_dicts(state_dicts):
#         avg_state = {}
#         for key in state_dicts[0]:
#             # print(f"Key: {key}, Shape: {state_dicts[0][key].shape}")
#             avg_state[key] = sum([sd[key] for sd in state_dicts]) / len(state_dicts)
#         return avg_state

#     # 合并 actor 和 critic 的 state dict
#     merged_actor_state = average_state_dicts(actor_states)
#     merged_critic_state = average_state_dicts(critic_states)

#     actor_save_path = os.path.join(buffer_dir, "actor_init.th")
#     critic_save_path = os.path.join(buffer_dir, "critic_init.th")

#     torch.save(merged_actor_state, actor_save_path)
#     torch.save(merged_critic_state, critic_save_path)

#     print(f"Merged actor saved to {actor_save_path}")
#     print(f"Merged critic saved to {critic_save_path}")

# 从child folder提取cls和policy在本函数中处理，run中只读取当前layer的文件夹
def train_merge_team(groups,
                     is_doe,
                     layer,
                     decompose_id,
                     group_id,
                     iter_id,
                     sample_id,
                     buffer_dir,
                     max_reward_code_path_for_each_group,
                     Time,
                     task_env,
                     suffix,
                     rl_runs
                     ):

    """
    
    首先合并child group信息，得到target task（当前任务）的role_ids和num agents
    生成本layer的config（考虑用create task替换）
    读取child group的doe cls，合并得到新的doe cls，存储在 merged_doe_name，用于rl训练开始时加载给mac learner
    todo：读取child group的policy pth，合并得到新的policy并存储到本target task下作为init policy

    执行run.py
        修改ckpt path不为""，以在训练初期 learner.load init team policy
        加载 merged doe name 这个cls，利用load模式的from config
        进行训练
        训练结束后存储buffer到本层folder
        todo：更改role_ids的list命名
        读取buffer进行新的cls训练，利用train模式的from config，存储为 save doe name，用于下一阶段训练
        存储final policy ckpt 到文件夹路径


    """


    team_structure = {
        "total_members": 0,
        "num_subteams": len(groups),
        "task_assignments": {}
    }

    # 记录当前的队员 ID
    current_id = 0

    # 遍历每个 group，将信息合并
    for group in groups:
        cur_group_id = group["group_number"] - 1
        num_agents = group["number_of_agents"]
        # 更新总成员数量
        team_structure["total_members"] += num_agents
        # 为每个任务分配队员 ID
        task_assignments = {
            "task": f"goal_{cur_group_id}",
            "member_ids": list(range(current_id, current_id + num_agents))
        }
        # 更新当前 ID
        current_id += num_agents
        # 将任务分配信息添加到队伍结构中
        team_structure["task_assignments"][f"group_{cur_group_id}"] = task_assignments

    """
    {
        "total_members": 8,
        "num_subteams": 2,
        "task_assignments": {
            "group_1": {
                "task": "goal_1",
                "member_ids": [0, 1, 2, 3, 4]
            },
            "group_2": {
                "task": "goal_2",
                "member_ids": [5, 6, 7]
            },
        }
    }
    """

    role_list = []
    # # 初始化任务 ID 计数器
    # task_id_counter = 0

    # 遍历每个子团队，提取任务信息
    for group_key, group_info in team_structure["task_assignments"].items():
        task_label = int(group_info["task"].split('_')[1])
        member_ids = group_info["member_ids"]
        # 为每个成员添加对应的任务 ID,使用其group id
        role_list.extend([task_label] * len(member_ids))

        # # 任务 ID 计数器加 1
        # task_id_counter += 1

    # role_list = [6, 6, 7]
    # role_list = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]，可用于指定merged doe的role ids
    # [attack attack defend]

    # 把团队角色信息转为role ids
    role_ids = {}
    for agent_id, role in enumerate(role_list):
        # task_name = list(team_structure["task_assignments"].values())[role]["task"]  # 获取子团队任务名称
        task_name = f"goal_{role}"
        if task_name not in role_ids:
            role_ids[task_name] = []
        role_ids[task_name].append(agent_id)
        # role_ids:{"goal_6": [0], "goal_7": [1]}



    """
    To LZH:
    这里需要考虑加相对路径，修改 template file path 的位置，以及template config name 可以换成ia2c，作为基础参数模版，可以用于训练非doe的
    """

    # 读取 ia2c_ns.yaml 作为模板,保持param non sharing，之前用的ia2c
    template_config_name = 'ippo_ns'
    template_file_path = f'{SRC_DIR}/config/algs/{template_config_name}.yaml'
    with open(template_file_path, 'r', encoding='utf-8') as template_file:
        template_data = yaml.safe_load(template_file)

    # 修改模板数据以生成 doe_ia2c.yaml 格式
    # template_data['mac'] = "doe_mac"  # 修改 mac
    template_data['mac'] = "non_shared_doe_mac"  # 使用ns doe mac
    template_data['target_update_interval_or_tau'] = 0.01  # 修改更新间隔
    template_data['learner'] = "doe_ippo_learner"  # 修改学习器
    template_data['entropy_coef'] = 0.01  # 修改熵系数
    template_data['use_rnn'] = True  # 使用 RNN
    # template_data['critic_type'] = "ac_critic"  # 修改评论家类型
    template_data['critic_type'] = "ac_critic_ns"  # 使用ns critic
    template_data['name'] = "doe_ippo_ns"  # 使用ns

    # 11111111111指定 merge 以后的 full team doe cls 存储名称
    # 0505更正：这里指定的是合并doe cls以后存储的文件，用于实验开始时load
    # merged_doe_name = f"doe_{template_config_name}_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}_merged"

    # 这里layer需要减1，因为从layer 2 的group 6+7合并成layer1的group5了，对应的scenarios也是layer1
    merged_doe_name = f"cls_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}_merged"


    # 0505更正：这里是load doe name，作用是本轮rl训练结束后，用当前的buffer训练新的doe cls存储位置
    # In multi-layer: add current iter and sample and this layer and this decomposed id to save for the father training
    save_current_layer_merged_doe_path = f"cls_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}"
    # save_current_layer_merged_doe_path = f"doe_{template_config_name}_layer{layer}_decomposition{decompose_id}_subtask{group_id}_iter{iter_id}_sample{sample_id}"

    # role_ids_normalized = normalize_keys(role_ids)

    """0505更新
    这里也可以考虑统一用create task来更新创建cfg？"""

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
        "obs_agent_id": True,
        "time_stamp": Time,
        "doe_type": "mlp",
        "ent_coef": 1.0,
        "base_lr": 1.0,
        "base_ent": 1.0,
        "boost_lr_coef": 1.0,
        "boost_ent_coef": 1.0,
        "checkpoint_path":buffer_dir,
        "doe_classifier_cfg": {
            "doe_type": "mlp",
            "load_mode": "train",
            "save_classifier": True,  # 首次训练没有doe，不用save，不过这里已经是merge阶段，而且使用doe，那么肯定要true
            "layer_tmp_dir": buffer_dir,
            "save_doe_name": f"{save_current_layer_merged_doe_path}.pt",
            "load_doe_name": f"{merged_doe_name}.pt",  # 用于训练 merge team 加载的原始 doe cls，直接 load 模式
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

    """
    这里save load doe name需要重新修改一下，目前是 'save_doe_name' =
    'cls_layer2_decomposition0_subtask8_iter0_sample0.pt'
    'load_doe_name' =
    'doe_ia2c_layer2_decomposition0_subtask8_iter0_sample0_merged.pt'
    尤其是buffer dir需要考虑重新命名以及指定到decomposition0，不要指定到group0-6，为了提取两个不同group的cls
    """

    # merge doe cls，保存到cfg.merge_doe_name
    # merge_cfg_doe_params = template_data["doe_classifier_cfg"]
    merge_doe_cls(groups, team_structure["total_members"], role_list, buffer_dir, merged_doe_name,
                  max_reward_code_path_for_each_group)


    # merge policy 需要根据non param share适配，在rnn_ns_agent.py中
    # self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])
    # 以这种list形式调用rnnagent创建list，所以只需要append再存储成一个actor就行
    merge_policy(groups, buffer_dir)

    # 还有一个问题是，在run里面有一个load model，但是那个只是load整个任务全部团队的？似乎需要在这里新增一个merge policy，
    # 合并存储为一个新的policy，命名为 init_team_policy.pth，训练结束后存储的是另外的，这样互不干涉

    
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
                full_process = subprocess.Popen(params)
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
                # full_process = subprocess.Popen(params, stdout=f, stderr=f)
                full_process = subprocess.Popen(params)

                full_process.wait()
            # Modified the check of successful training
            # block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(full_process)
    # 如果不使用doe训练（但也不是最底层任务）
    else:
        rl_logpath = f"{OUTPUT_DIR}/{task_env}{suffix}_full_training.txt"
        with open(rl_logpath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={template_config_name}',
                f'--env-config={task_env}',
            ]
            full_process = subprocess.Popen(params)
            full_process.wait()
        # block_until_training(rl_logpath, log_status=True, iter_num=iter, response_id=response_id)

    full_rl_training_performance = []
    full_rl_training_performance.append(full_process)
    # 似乎也不用save一个performance，tensorboard会自动生成的，就是找起来麻烦，要考虑一下logger的file合并
    # save(full_rl_training_performance)

    print('Merged Full Training Has Finished')
    return rl_runs



if __name__ == "__main__":
    # main(model="gpt-4-turbo", n_decomposition=1, n_reward=5, temperature=1, task_env="gfootball", alg_cfg="ia2c",
    #      use_doe=False, n_improve_iter=2)

    response_id = 0
    layer = 2  # 这个layer对应的是child_layer
    merge_layer = layer-1   # 似乎没用到target选项

    # group_id = 6
    merge_group_id = 2
    child_group_id = [6, 7]

    response_r_id = 0
    n_agents = 1
    merge_n_agents = 2
    suffix = "_GPT"
    alg_cfg = "ippo_ns"   # 之前用ia2c，替换成non sharing
    task_env = "gfootball"
    rl_runs = []
    Time = "2025-05-14-08-15-41_2222"

    TIMEOUT = 20
    # 如果是最底层，不用doe

    # 为了debug，暂时关掉
    for group_id in child_group_id:
        logging.info(
            f"Training for Decomposition {response_id} Layer{layer} Group{group_id} ")
        # Create Task YAML file
        create_task(CONFIG_ENVS_DIR, task_env, layer, response_id, response_r_id, n_agents,
                    group_id, iter=0, suffix=suffix)

        # 首层训练不用带doe，添加额外参数
        create_train_cfg(CONFIG_ALGS_DIR, Time, alg_cfg, layer, response_id, response_r_id,
                         n_agents, group_id, iter=0, task_env=task_env, results_dir=RESULTS_DIR, init_layer=True)

        # Execute the python file with flags
        rl_filepath = f"{OUTPUT_DIR}/gfootball{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter1_sample{response_r_id}.txt"


        with open(rl_filepath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={alg_cfg}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter1_sample{response_r_id}',
                f'--env-config={task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter1_sample{response_r_id}',
            ]
            # import sys
            # sys.path.append('/data/qiaodan/projects/GRF_SUBTASK/gfootball')
            # 底层任务需要修改这个Popen
            # process = subprocess.Popen(params)
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




    ##############Use doe for training##############
    use_doe = True
    # 创建 merge target team 的相关task
    # 暂时创建一个group5，没搞清楚这里什么原因，可能是那个group id-1导致的
    # merge_group_id_1 = merge_group_id - 1


    # 这个似乎是GPT创建reward的循环，暂时不用？
    # 但是trian config for merged group5在哪儿
    create_task(CONFIG_ENVS_DIR, task_env, merge_layer, response_id, response_r_id,
                merge_n_agents,
                group_id=merge_group_id, iter=0, suffix=suffix)

    # reward = checkpoint, maps=5_vs_5

    child_tasks = [
        {"group_number": 7, "number_of_agents": 1},
        {"group_number": 8, "number_of_agents": 1},
    ]

    # 原来是 group6 group7，感觉不太对劲，也许需要改成group 7 8？代表reward func
    max_reward_code_path_for_each_group={"group6": "reward_layer2_decomposition0_subtask6_iter1_sample0.py",
                                         "group7":"reward_layer2_decomposition0_subtask7_iter1_sample0.py"}

    # 指定本层merge的所有存储文件路径为：

    save_dir=f'{MEDoE_DIR}/doe_epymarl-main/results/gfootball/{Time}/decomposition{response_id}/group{merge_group_id}'
    os.makedirs(save_dir, exist_ok=True)
    # 该路径作为buffer_dir传入train_merge_team函数，save as "doe_classifier_cfg".load_doe_buffer_path
    # 并传入 doe_classifier_config_loader["load"]作为path，load_doe_name 是经过merge处理以后的新doe cls
    # doe cls 加载路径为 absolute_path = os.path.join(buffer_path, cfg["load_doe_name"])


    # 这里layer是否需要target？还是新增一个属性？
    rl_runs = train_merge_team(child_tasks, use_doe, layer=merge_layer, decompose_id=response_id,
                               group_id=merge_group_id, iter_id=1, sample_id=response_r_id,
                               buffer_dir=save_dir,
                               max_reward_code_path_for_each_group=max_reward_code_path_for_each_group,
                               Time=Time, task_env=task_env, suffix=suffix, rl_runs=rl_runs)


#########################################################################################
# if __name__ == "__main__":
#     # main(model="gpt-4-turbo", n_decomposition=1, n_reward=5, temperature=1, task_env="gfootball", alg_cfg="ia2c",
#     #      use_doe=False, n_improve_iter=2)
#
#     response_id = 0
#     layer = 2  # 这个layer对应的是child_layer
#     merge_layer = layer - 1  # 似乎没用到target选项
#
#     # group_id = 6
#     merge_group_id = 5
#     child_group_id = [6, 7]
#
#     response_r_id = 0
#     n_agents = 1
#     merge_n_agents = 2
#     suffix = "_GPT"
#     alg_cfg = "ippo_ns"  # 之前用ia2c，替换成non sharing
#     task_env = "gfootball"
#     rl_runs = []
#     Time = "0513_ia2c_ns_1"
#
#     TIMEOUT = 20
#     # 如果是最底层，不用doe
#
#     # 为了debug，暂时关掉
#     for group_id in child_group_id:
#         logging.info(
#             f"Training for Decomposition {response_id} Layer{layer} Group{group_id} ")
#         # Create Task YAML file
#         create_task(CONFIG_ENVS_DIR, task_env, layer, response_id, response_r_id, n_agents,
#                     group_id, iter=0, suffix=suffix)
#
#         # 首层训练不用带doe，添加额外参数
#         create_train_cfg(CONFIG_ALGS_DIR, Time, alg_cfg, layer, response_id, response_r_id,
#                          n_agents, group_id, iter=0, task_env=task_env, results_dir=RESULTS_DIR, init_layer=True)
#
#         # Execute the python file with flags
#         rl_filepath = f"{OUTPUT_DIR}/gfootball{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{0}_sample{response_r_id}.txt"
#
#
#         with open(rl_filepath, 'w') as f:
#             script_path = f'{SRC_DIR}/main.py'
#             params = [
#                 'python', '-u', script_path,
#                 f'--config={alg_cfg}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{0}_sample{response_r_id}',
#                 f'--env-config={task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{0}_sample{response_r_id}',
#             ]
#             # import sys
#             # sys.path.append('/data/qiaodan/projects/GRF_SUBTASK/gfootball')
#             # 底层任务需要修改这个Popen
#             # process = subprocess.Popen(params)
#             process = subprocess.Popen(params, stdout=f, stderr=f)
#
#             # 获取文件的初始修改时间
#             while True:
#                 initial_mtime = os.path.getmtime(rl_filepath)
#                 initial_mtime = datetime.datetime.fromtimestamp(initial_mtime)  # 时间转为datetime格式
#                 start_time = datetime.datetime.now()
#                 delta_time = start_time - initial_mtime  # 时间差
#                 delta_seconds = delta_time.total_seconds()  # 时间差转成秒
#                 if delta_seconds > TIMEOUT:  # 如果文件更新时间大于30秒，重新启动程序
#                     print(
#                         f"Overtime：It seems that the training is stuck or finished, subprocess terminates")
#                     process.kill()  # 终止子进程
#                     break
#                 # while process.poll() is None:  # 检查子进程是否还在运行
#                 #     # 检查文件的最后修改时间
#                 #     current_mtime = os.path.getmtime(rl_filepath)
#                 #     # 如果文件超过了 1 分钟没有更新
#                 #     if current_mtime == initial_mtime and (time.time() - start_time) > TIMEOUT:
#                 #         print(f"Overtime：It seems that the training is stuck, subprocess terminates")
#                 #         process.terminate()  # 终止子进程
#                 #         break
#                 # 等待一段时间后再检查
#                 time.sleep(1)
#
#             process.wait()
#         # Modified the check of successful training
#         # block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
#         rl_runs.append(process)
#
#     ##############Use doe for training##############
#     use_doe = True
#     # 创建 merge target team 的相关task
#     # 暂时创建一个group5，没搞清楚这里什么原因，可能是那个group id-1导致的
#     # merge_group_id_1 = merge_group_id - 1
#
#     # 这个似乎是GPT创建reward的循环，暂时不用？
#     # 但是trian config for merged group5在哪儿
#     create_task(CONFIG_ENVS_DIR, task_env, merge_layer, response_id, response_r_id,
#                 merge_n_agents,
#                 group_id=merge_group_id, iter=0, suffix=suffix)
#
#     # reward = checkpoint, maps=5_vs_5
#
#     child_tasks = [
#         {"group_number": 7, "number_of_agents": 1},
#         {"group_number": 8, "number_of_agents": 1},
#     ]
#
#     # 原来是 group6 group7，感觉不太对劲，也许需要改成group 7 8？代表reward func
#     max_reward_code_path_for_each_group = {"group6": "reward_layer2_decomposition0_subtask6_iter0_sample0.py",
#                                            "group7": "reward_layer2_decomposition0_subtask7_iter0_sample0.py"}
#
#     # 指定本层merge的所有存储文件路径为：
#
#     save_dir = f'{MEDoE_DIR}/doe_epymarl-main/results/gfootball/{Time}/decomposition{response_id}/group{merge_group_id}'
#     os.makedirs(save_dir, exist_ok=True)
#     # 该路径作为buffer_dir传入train_merge_team函数，save as "doe_classifier_cfg".load_doe_buffer_path
#     # 并传入 doe_classifier_config_loader["load"]作为path，load_doe_name 是经过merge处理以后的新doe cls
#     # doe cls 加载路径为 absolute_path = os.path.join(buffer_path, cfg["load_doe_name"])
#
#     # 这里layer是否需要target？还是新增一个属性？
#     rl_runs = train_merge_team(child_tasks, use_doe, layer=merge_layer, decompose_id=response_id,
#                                group_id=merge_group_id, iter_id=0, sample_id=response_r_id,
#                                buffer_dir=save_dir,
#                                max_reward_code_path_for_each_group=max_reward_code_path_for_each_group,
#                                Time=Time, task_env=task_env, suffix=suffix, rl_runs=rl_runs)
