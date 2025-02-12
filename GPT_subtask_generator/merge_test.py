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
FOOTBALL_MEDoE_DIR = os.path.abspath(os.path.join(current_dir, "../../../"))

SRC_DIR = os.path.join(FOOTBALL_MEDoE_DIR, "doe_epymarl-main/src")
sys.path.append(SRC_DIR)

from run import *

client = OpenAI(api_key="KEY")
ROOT_DIR = os.getcwd()
parent_dir = os.path.dirname(ROOT_DIR)
GFOOTBALL_DIR = os.path.dirname(parent_dir)
CONFIG_ENVS_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/src/config/envs')
CONFIG_ALGS_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/src/config/algs')
MAP_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/maps')
REWARD_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/rewards')
prompt_dir = f'{ROOT_DIR}/utils/prompts'
logging.basicConfig(level=logging.INFO)

Time = datetime.datetime.now()
Time = Time.strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR = f"outputs/{Time}"
MAP_DIR = f"{MAP_DIR}/{Time}"
REWARD_DIR = f"{REWARD_DIR}/{Time}"
SACRED_DIR = os.path.join(FOOTBALL_MEDoE_DIR, "doe_epymarl-main/results/sacred")
TENSORBOARD_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/results/tb_logs')
# GRF_SCENARIO_DIR = f"{GFOOTBALL_DIR}/scenarios/{Time}"

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
        max_reward_code_path = max_reward_code_path_for_each_group[f"group{group_id}"].replace("reward", "cls").replace(".py", ".pt")
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

def train_merge_team(groups, is_doe, layer, decompose_id, buffer_dir, max_reward_code_path_for_each_group, Time):

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
    template_config_name = 'ia2c'
    template_file_path = f'{SRC_DIR}/config/algs/{template_config_name}.yaml'
    with open(template_file_path, 'r', encoding='utf-8') as template_file:
        template_data = yaml.safe_load(template_file)

    # 修改模板数据以生成 doe_ia2c.yaml 格式
    template_data['mac'] = "doe_mac"  # 修改 mac
    template_data['target_update_interval_or_tau'] = 0.01  # 修改更新间隔
    template_data['learner'] = "doe_ia2c_learner"  # 修改学习器
    template_data['entropy_coef'] = 0.01  # 修改熵系数
    template_data['use_rnn'] = True  # 使用 RNN
    template_data['critic_type'] = "ac_critic"  # 修改评论家类型
    template_data['name'] = "doe_ia2c"  # 修改名称

    # 11111111111指定 merge 以后的 full team doe cls 存储名称
    merged_doe_name = f"doe_{template_config_name}_layer{layer}_decomposition{decompose_id}_merged"

    # In multi-layer: add current iter and sample and this layer and this decomposed id to save for the father training
    save_current_layer_merged_doe_path = f"merged_doe_buffer"

    # role_ids_normalized = normalize_keys(role_ids)

    # 添加 DoE 相关参数
    doe_params = {
        # In multi-layer: add current iter and sample and this layer and this decomposed id to save for the father training
        "layer_id": "target",
        "decomposition_id": decompose_id,
        "group_id": "target",
        "iter_id": "target",
        "sample_id": "target",
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
    merged_doe_config_name = f"doe_ia2c_layer{layer}_decomposition{decompose_id}_merged"
    new_yaml_file_path = f'{FOOTBALL_MEDoE_DIR}/doe_epymarl-main/src/config/algs/{merged_doe_config_name}.yaml'

    # 将修改后的数据写入新的 YAML 文件
    with open(new_yaml_file_path, 'w', encoding='utf-8') as new_yaml_file:
        yaml.dump(template_data, new_yaml_file, allow_unicode=True)

    print(f"New DOE YAML File {merged_doe_config_name}")

    # merge doe cls，保存到cfg.merge_doe_name
    merge_cfg_doe_params = template_data["doe_classifier_cfg"]
    merge_doe_cls(groups, team_structure["total_members"], role_list, buffer_dir, merged_doe_name, max_reward_code_path_for_each_group)


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
    origin_env_config = "gfootball"

    if is_doe:
        print("1111111111111111111111111111111")
        rl_logpath = f"full_training_depth_1_{origin_env_config}_doe.txt"
        print(f'--config={merged_doe_config_name}')
        print(f'--env-config={origin_env_config}')
        with open(rl_logpath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={merged_doe_config_name}',
                f'--env-config={origin_env_config}',
            ]
            full_process = subprocess.Popen(params, stdout=f, stderr=f)
            full_process.wait()
        # block_until_training(rl_logpath, log_status=True, iter_num=iter, response_id=response_id)
    else:
        rl_logpath = f"full_training_depth_1_{origin_env_config}.txt"
        with open(rl_logpath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={template_config_name}',
                f'--env-config={origin_env_config}',
            ]
            full_process = subprocess.Popen(params, stdout=f, stderr=f)
            full_process.wait()
        # block_until_training(rl_logpath, log_status=True, iter_num=iter, response_id=response_id)

    full_rl_training_performance = []
    full_rl_training_performance.append(full_process)
    # 似乎也不用save一个performance，tensorboard会自动生成的，就是找起来麻烦，要考虑一下logger的file合并
    # save(full_rl_training_performance)

    print('Merged Full Training Has Finished')



workspace_dir = Path.cwd()
logging.info(f"Workspace: {workspace_dir}")
logging.info(f"Project Root: {ROOT_DIR}")


    # 完成了第一个子任务 reward 生成

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
groups = []
group_info_1 = {
            "group_number": 1,
            "number_of_agents": 2,
            "training_goal": "Mastering ball control and development of passing accuracy. This group will focus on maintaining possession by effectively using the dribble, executing short and long passes, and responding defensively when required. Key actions to train include Dribble, Stop-Dribble, Short Pass, Long Pass, and defensive maneuvres like Sliding."
        }
group_info_2 = {
    "group_number": 2,
    "number_of_agents": 3,
    "training_goal": "Focused on creating and exploiting scoring opportunities. Agents in this group should learn to optimally position themselves, decide when to engage in sprints to outmaneuver opponents, and perform effective shooting. Primary actions include Shot, Sprint, Stop-Sprint, High Pass (for crosses), and mastering the timing for offensive moves."
}
groups.append(group_info_1)
groups.append(group_info_2)
max_reward_code_path_for_each_group={"group0":'reward_layer0_decomposition0_subtask0_iter0_sample0.py',"group1":'reward_layer0_decomposition0_subtask1_iter0_sample0.py'}
train_merge_team(groups, use_doe, layer=0, decompose_id=0, buffer_dir=f'{FOOTBALL_MEDoE_DIR}/doe_epymarl-main/results/buffers/gfootball/2024-11-27-01-31-39', max_reward_code_path_for_each_group=max_reward_code_path_for_each_group, Time='2024-11-27-01-31-39')





