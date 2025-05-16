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
from utils.rag import *
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

TOGETHER_API_KEY = "key"

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


# # 处理长文本，确保生成的 YAML 不包含复杂键
# def normalize_keys(data):
#     if isinstance(data, dict):
#         return {str(k): normalize_keys(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [normalize_keys(i) for i in data]
#     else:
#         return data

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
    if layer == "target":
        template_data['t_max'] = 20050000


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
        "checkpoint_path": buffer_dir,
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
                # full_process = subprocess.Popen(params, stdout=f, stderr=f)
                full_process = subprocess.Popen(params, stdout=f, stderr=f)

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




def main(model, n_decomposition, temperature, task_env, alg_cfg, use_doe, n_improve_iter):
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
    initial_system_evaluate = file_to_string(f'{prompt_dir}/RAG/initial_system_improve.txt')
    initial_user_evaluate = file_to_string(f'{prompt_dir}/RAG/initial_user_improve.txt')

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

    main_task = "learn to play a 5 vs 5 football game"
    num_agents = 5
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
                # reply_scenario = response_scenario_cur.choices[0].message.content

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
                exec_success = False
                while not exec_success:
                    # Reward generation and improving:
                    logging.info(
                        f"Rewards Generation: Generating Reward function for Decomposition {response_id} Layer{layer} Group{group_id} with {model}")

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


                    max_reward_code_path_for_each_group[f'group{group_id}'] = None
                    iter_id = 0
                    chunk_size_r = 1
                    rl_runs = []
                    max_attempts_r = 1000

                    # n reward 为 1

                    while attempt < max_attempts_r:
                        print("ATTEMPT:", attempt)
                        try:
                            reply_rewards_cur = client.chat.completions.create(model=model,
                                                                               messages=cur_messages_r,
                                                                               temperature=temperature,
                                                                               n=chunk_size_r)
                            break
                        except Exception as e:
                            logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                            time.sleep(1)  # 等待 1 秒后重试
                            attempt += 1

                    if reply_rewards_cur is None:
                        logging.info("Code terminated due to too many failed attempts!")
                        exit()

                    reply_reward = reply_rewards_cur.choices[0].message.content
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

                    # response id是分解几层，response r id是sample的reward function个数
                    with open(
                            f"{REWARD_DIR}/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.py",
                            'w') as file:
                        file.writelines("import gym" + '\n')
                        file.writelines("import numpy as np" + '\n')
                        file.writelines(reward_code_string + '\n')

                    with open(
                            f"{OUTPUT_DIR}/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.py",
                            'w') as file:
                        file.writelines("import gym" + '\n')
                        file.writelines("import numpy as np" + '\n')
                        file.writelines(reward_code_string + '\n')

                    # Save the reward function in the GRF Env
                    with open(
                            f"{GFOOTBALL_DIR}/rewards/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.py",
                            'w') as file:
                        file.writelines("import gym" + '\n')
                        file.writelines("import numpy as np" + '\n')
                        file.writelines(reward_code_string + '\n')


                    ######################## RAG Evaluation ###############################################
                    knowledge_base_embeddings = np.load('RAG_data/knowledge_base_embeddings.npy')
                    with open('RAG_data/merged_knowledge_base.json', 'r') as file:
                        knowledge_base = json.load(file)

                    for improve_id in range(n_improve_iter):
                        eval_reward = reward_code_string
                        eval_training_goal = task['training_goal']
                        eval_num_agents = task['number_of_agents']
                        query = f"Training goal: {eval_training_goal}. Reward function: {eval_reward}."
                        query_embedding = generate_embeddings([query], 'togethercomputer/m2-bert-80M-2k-retrieval', TOGETHER_API_KEY)[0]
                        similarity_scores = cosine_similarity([query_embedding], knowledge_base_embeddings)
                        indices = np.argsort(-similarity_scores)
                        top_10_sorted_suggestions = [knowledge_base[index]['suggestions'] for index in indices[0]][:10]
                        top_10_data = [knowledge_base[index] for index in indices[0][:10]]
                        with open(f'{OUTPUT_DIR}/top10_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.json', 'w', encoding='utf-8') as f:
                            json.dump(top_10_data, f, ensure_ascii=False, indent=2)

                        top10_input = ""
                        for i, entry in enumerate(top_10_data):
                            training_goal = entry.get('training_goal', '')
                            num_agents = entry.get('num_agents', '')
                            reward_function = entry.get('reward_function', '')
                            component = entry.get('component', '')
                            evaluation = entry.get('reward_function', '')
                            suggestions = entry.get('suggestions', [])
                            # 构建 GPT 输入格式
                            formatted_input = f"---\nEntry {i + 1}:\nTraining goal: {training_goal}\nNum_agents: {num_agents}\nReward function:\n{reward_function}\nComponent:\n{component}\nEvaluation:\n{evaluation}\nSuggestions: {', '.join(suggestions)}\n"

                            # 将每个条目的格式化文本追加到 final_input
                            top10_input += formatted_input

                        curr_code_output_tip_rewards_eva = code_output_tip_rewards.format(number_of_agents=eval_num_agents,
                                                                                      example_of_o=example_of_o,
                                                                                      reward_signature=reward_signature)
                        cur_initial_system_evaluate = initial_system_evaluate + example_rewards + curr_code_output_tip_rewards_eva
                        cur_initial_user_evaluate = initial_user_evaluate.format(training_goal=eval_training_goal,
                                                               num_agents=eval_num_agents,
                                                               reward_function=eval_reward,
                                                               top_10=top10_input,
                                                               env_code=env_code)
                        messages_evaluate = [{"role": "system", "content": cur_initial_system_evaluate},
                                             {"role": "user", "content": cur_initial_user_evaluate}]
                        response_cur_e = None
                        attempt = 0
                        max_attempts = 100  # 限制最大尝试次数
                        chunk_size_e = 1  # 只生成 1 个回复

                        while attempt < max_attempts:
                            print("ATTEMPT:", attempt)
                            try:
                                response_cur_e = client.chat.completions.create(
                                    model="gpt-4-turbo",
                                    messages=messages_evaluate,
                                    temperature=1,
                                    n=chunk_size_e
                                )
                                break  # 成功则跳出循环
                            except Exception as e:
                                logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                                time.sleep(1)  # 等待 1 秒后重试
                                attempt += 1

                        # 如果所有尝试都失败，则终止
                        if response_cur_e is None:
                            logging.error("Failed to generate response after multiple attempts!")
                            exit()

                        iter_id += 1
                        # 解析返回的响应
                        response_e = response_cur_e.choices[0].message.content  # 只取第一个回复
                        print("Generated Response:", response_e)

                        # Regex patterns to extract python code enclosed in GPT response
                        patterns = [
                            r'```python(.*?)```',
                            r'```(.*?)```',
                            r'"""(.*?)"""',
                            r'""(.*?)""',
                            r'"(.*?)"',
                        ]
                        for pattern in patterns:
                            reward_code_string = re.search(pattern, response_e, re.DOTALL)
                            if reward_code_string is not None:
                                reward_code_string = reward_code_string.group(1).strip()
                                break
                        reward_code_string = response_e if not reward_code_string else reward_code_string

                        print("Reward Code String 1:", reward_code_string)

                        # Remove unnecessary imports
                        lines = reward_code_string.split("\n")
                        for i, line in enumerate(lines):
                            if line.strip().startswith("class "):
                                reward_code_string = "\n".join(lines[i:])

                        print("Reward Code String 2:", reward_code_string)

                        # response id是分解几层，response r id是sample的reward function个数
                        with open(
                                f"{REWARD_DIR}/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.py",
                                'w') as file:
                            file.writelines("import gym" + '\n')
                            file.writelines("import numpy as np" + '\n')
                            file.writelines(reward_code_string + '\n')

                        with open(
                                f"{OUTPUT_DIR}/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.py",
                                'w') as file:
                            file.writelines("import gym" + '\n')
                            file.writelines("import numpy as np" + '\n')
                            file.writelines(reward_code_string + '\n')

                        # Save the reward function in the GRF Env
                        with open(
                                f"{GFOOTBALL_DIR}/rewards/reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.py",
                                'w') as file:
                            file.writelines("import gym" + '\n')
                            file.writelines("import numpy as np" + '\n')
                            file.writelines(reward_code_string + '\n')
                    ################Finsh refining and get final reward function
                    def has_subtasks(task_tree, group_number):
                        for layer in task_tree.values():
                            for task in layer:
                                if task.get("father_group_number") == group_number:
                                    return True
                        return False

                    ################################ Train ############################################################
                    TIMEOUT = 30
                    #如果是最底层，不用doe
                    if not has_subtasks(task_tree, task["group_number"]):
                        # Create Task YAML files
                        create_task(CONFIG_ENVS_DIR, task_env, layer, response_id, 0, task['number_of_agents'],
                                    task['group_number'] - 1, iter_id, suffix)
                        create_train_cfg(CONFIG_ALGS_DIR, Time, alg_cfg, layer, response_id, 0,
                                         task['number_of_agents'], task['group_number'] - 1, iter_id, task_env=task_env, results_dir=RESULTS_DIR, init_layer=True)
                        # Execute the python file with flags
                        rl_filepath = f"{OUTPUT_DIR}/{task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.txt"
                        with open(rl_filepath, 'w') as f:
                            script_path = f'{SRC_DIR}/main.py'
                            params = [
                                'python', '-u', script_path,
                                f'--config={alg_cfg}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0',
                                f'--env-config={task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0',
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
                        create_task(CONFIG_ENVS_DIR, task_env, layer, response_id, 0,
                                    task['number_of_agents'],
                                    task['group_number'] - 1, iter_id, suffix)
                        child_tasks = get_child_tasks(task_tree, layer, group_id + 1)
                        buffer_dir = f'{MEDoE_DIR}/doe_epymarl-main/results/gfootball/{Time}/decomposition{response_id}/group{group_id}'
                        os.makedirs(buffer_dir, exist_ok=True)
                        rl_runs = train_merge_team(child_tasks, use_doe, layer=layer, decompose_id=response_id,
                                         group_id = group_id, iter_id = iter_id, sample_id = 0,
                                         buffer_dir=buffer_dir,
                                         max_reward_code_path_for_each_group=max_reward_code_path_for_each_group,
                                         Time=Time, task_env=task_env, suffix=suffix, rl_runs = rl_runs)

                    # Check execution:
                    content = ''
                    result_path = f'{SACRED_DIR}/{Time}/scenario_layer{layer}_decomposition{response_id}_subtask{group_id}/scoring, reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0/1'
                    rl_filepath = f"{OUTPUT_DIR}/{task_env}{suffix}_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.txt"
                    try:
                        with open(rl_filepath, 'r') as f:
                            stdout_str = f.read()
                    except:
                        # content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                        content += execution_error_feedback.format(
                            traceback_msg="Code Run cannot be executed because reward class is wrongly formatted! Please re-write an entirely new reward function!")
                        continue
                    traceback_msg = filter_traceback(stdout_str)
                    done_string = 'absl Dump "episode_done": count limit reached / disabled'
                    num_done_string = stdout_str.count(done_string)
                    exec_success = False
                    if traceback_msg == '':
                        if num_done_string < 6:
                            print("Wrong scenario!")
                            content += execution_error_feedback.format(
                                traceback_msg="Wrong Scenario without Goalkeeper")
                        else:
                            exec_success = True
                            max_reward_code_path_for_each_group[
                                f'group{group_id}'] = f"reward_layer{layer}_decomposition{response_id}_subtask{group_id}_iter{iter_id}_sample0.py"

                    else:
                        print("Spotting errors in the reward function! Regenerating...")
                        # Otherwise, provide execution traceback error feedback
                        content += execution_error_feedback.format(traceback_msg=traceback_msg)


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
        target_buffer_dir=f'{MEDoE_DIR}/doe_epymarl-main/results/gfootball/{Time}/decomposition{response_id}/grouptarget'
        os.makedirs(target_buffer_dir, exist_ok=True)
        rl_runs = train_merge_team(task_tree[0], use_doe, layer="target", decompose_id=response_id, group_id = "target", iter_id = "target",
                                   sample_id = "target", buffer_dir=target_buffer_dir,
                                   max_reward_code_path_for_each_group=max_reward_code_path_for_each_group, Time=Time,
                                   task_env = task_env, suffix = suffix, rl_runs = [])

    # 完成了所有方案 n decomposition plan 的任务生成，Execute the Main task using w/w. DOE:



if __name__ == "__main__":
    main(model="gpt-4-turbo", n_decomposition=1, temperature=1, task_env="gfootball", alg_cfg="ippo_ns",
         use_doe=False, n_improve_iter=3)