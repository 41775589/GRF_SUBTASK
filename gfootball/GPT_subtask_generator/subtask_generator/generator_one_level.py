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
from utils.create_task import create_task
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
CONFIG_ROOT_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/src/config/envs')
MAP_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/maps')
REWARD_DIR = os.path.join(FOOTBALL_MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/rewards')
prompt_dir = f'{ROOT_DIR}/utils/prompts'
logging.basicConfig(level=logging.INFO)

Time = datetime.datetime.now()
Time = Time.strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR = f"outputs/{Time}"
MAP_DIR = f"{MAP_DIR}/{Time}"
REWARD_DIR = f"{REWARD_DIR}/{Time}"
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
    

def merge_doe_cls(groups, n_agents, role_list, doe_path, merge_doe_name):
    # 初始化合并后的分类器
    merged_classifier = None
    merge_id = 0


    # 遍历每个组，加载对应的 DoE 分类器
    for group in groups:
        # 构建文件路径, 0_classifier.pt
        group_id = group["group_number"]
        classifier_path = f"{doe_path}/{group_id}_classifier.pt"
        
        # 加载分类器
        classifier_i = torch.load(classifier_path)

        # 创建初始化一个 merged cls，因为load可以直接加载原来的类的所有属性，我们只需要扩展classifier_i的mlps尺寸，更新 self.n_agents即可
        # 避免重新指定各种网络参数
        if merged_classifier is None:
            # merged_classifier = doe_classifier_config_loader(n_agents, merge_cfg, doe_path, load_mode='merge')
            merged_classifier = copy.deepcopy(classifier_i)
            merged_classifier.n_agents = n_agents
            merged_classifier.role_list = role_list
            # 扩展 lr 和 mlps 的数量
            merged_classifier.learning_rates = [merged_classifier.learning_rates[0]] * n_agents
            merged_classifier.mlps = [merged_classifier.mlps[0]] * n_agents

        # # 确保当前分类器的 mlps 列表长度与合并后的代理数量一致
        # assert classifier.n_agents == len(classifier1.mlps) + len(classifier2.mlps)

        # 合并历史分类器的参数到当前分类器中
        for doe_i in classifier_i.mlps:
            merged_classifier.mlps[merge_id].load_state_dict(doe_i.state_dict())
            merge_id += 1

    assert merge_id == n_agents-1
    # 保存合并后的分类器
    torch.save(merged_classifier, f'{doe_path}/{merge_doe_name}.pt')



def train_merge_team(groups, is_doe, decompose_id, buffer_dir):

    team_structure = {
        "total_members": 0,
        "num_subteams": len(groups),
        "task_assignments": {}
    }

    # 记录当前的队员 ID
    current_id = 0

    # 遍历每个 group，将信息合并
    for group in groups:
        group_id = group["group_number"]
        num_agents = group["number_of_agents"]
        
        # 更新总成员数量
        team_structure["total_members"] += num_agents
        
        # 为每个任务分配队员 ID
        task_assignments = {
            "task": group["training_goal"],
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
    template_file_path = f'GRF_SUBTASK/doe_epymarl-main/src/config/algs/{template_config_name}.yaml'
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

    # 指定 merge 以后的 full team doe cls 存储名称
    merged_doe_name = 'doe_ia2c_merge'
    save_current_layer_merged_doe_path = 'full_team_doe'

    # 添加 DoE 相关参数
    doe_params = {
        "use_doe": True,
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
    merged_doe_config_name = "doe_ia2c_plan_{}_merged".format(decompose_id)
    new_yaml_file_path = 'GRF_SUBTASK/doe_epymarl-main/src/config/algs/{}.yaml'.format(merged_doe_config_name)

    # 将修改后的数据写入新的 YAML 文件
    with open(new_yaml_file_path, 'w', encoding='utf-8') as new_yaml_file:
        yaml.dump(template_data, new_yaml_file, allow_unicode=True)

    print(f"New DOE YAML File {merged_doe_config_name}")

    # merge doe cls，保存到cfg.merge_doe_name
    merge_cfg_doe_params = template_data["doe_classifier_cfg"]
    merge_doe_cls(groups, team_structure["total_members"], role_list, buffer_dir, merged_doe_name)


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
        rl_logpath = f"full_training_depth_1_{origin_env_config}_doe.txt"
        with open(rl_logpath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={merged_doe_config_name}',
                f'--env-config={origin_env_config}',
                '--is_doe=True'
            ]
            full_process = subprocess.Popen(params, stdout=f, stderr=f)
        # block_until_training(rl_logpath, log_status=True, iter_num=iter, response_id=response_id)
    else:
        rl_logpath = f"full_training_depth_1_{origin_env_config}.txt"
        with open(rl_logpath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={template_config_name}',
                f'--env-config={origin_env_config}',
                '--is_doe=False'
            ]
            full_process = subprocess.Popen(params, stdout=f, stderr=f)
        # block_until_training(rl_logpath, log_status=True, iter_num=iter, response_id=response_id)

    full_rl_training_performance = []
    full_rl_training_performance.append(full_process)
    # 似乎也不用save一个performance，tensorboard会自动生成的，就是找起来麻烦，要考虑一下logger的file合并
    # save(full_rl_training_performance)

    print('Merged Full Training Has Finished')




def main(model, n_decomposition, n_reward, temperature, task, alg_cfg, use_doe, n_improve_iter):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    suffix = "_GPT"
    logging.info(f"Using LLM: {model}")

    # env_init = f'{ROOT_DIR}/env_code/__init__.py'
    env = f'{ROOT_DIR}/env_code/football_env.py'
    env_core = f'{ROOT_DIR}/env_code/football_env_core.py'
    # action_set= f'{ROOT_DIR}/env_code/football_action_set.py'
    observation_processor = f'{ROOT_DIR}/env_code/observation_processor.py'
    scenario_builder = f'{ROOT_DIR}/env_code/scenario_builder.py'
    reward_wrapper_example = f'{ROOT_DIR}/env_code/reward_wrapper_example.py'
    obs_o = f'{ROOT_DIR}/env_code/obs_o.py'
    # config = f'{ROOT_DIR}/env_code/config.py'
    wrappers = f'{ROOT_DIR}/env_code/wrappers.py'

    # env_init_code_string  = file_to_string(env_init)
    env_code_string = file_to_string(env)
    env_core_code_string = file_to_string(env_core)
    observation_processor_code_string = file_to_string(observation_processor)

    env_code = env_code_string + env_core_code_string + observation_processor_code_string

    scenario_builder_code_string = file_to_string(scenario_builder)
    wrappers = file_to_string(wrappers)

    reward_wrapper_example = file_to_string(reward_wrapper_example)
    obs_o = file_to_string(obs_o)

    three_vs_one_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_3_vs_1_with_keeper.py'
    corner = f'{ROOT_DIR}/scenario_examples/academy_corner.py'
    counterattack_easy = f'{ROOT_DIR}/scenario_examples/academy_counterattack_easy.py'
    counterattack_hard = f'{ROOT_DIR}/scenario_examples/academy_counterattack_hard.py'
    empty_goal = f'{ROOT_DIR}/scenario_examples/academy_empty_goal.py'
    empty_goal_close = f'{ROOT_DIR}/scenario_examples/academy_empty_goal_close.py'
    pass_and_shoot_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_pass_and_shoot_with_keeper.py'
    run_pass_and_shoot_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_run_pass_and_shoot_with_keeper.py'
    run_to_score = f'{ROOT_DIR}/scenario_examples/academy_run_to_score.py'
    run_to_score_with_keeper = f'{ROOT_DIR}/scenario_examples/academy_run_to_score_with_keeper.py'
    single_goal_versus_lazy = f'{ROOT_DIR}/scenario_examples/academy_single_goal_versus_lazy.py'
    five_vs_five = f'{ROOT_DIR}/scenario_examples/5_vs_5.py'

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
    initial_system_get_decomposition = file_to_string(f'{prompt_dir}/initial_system_get_decomposition.txt')
    initial_user_get_decomposition = file_to_string(f'{prompt_dir}/initial_user_get_decomposition.txt')
    initial_system_scenarios = file_to_string(f'{prompt_dir}/initial_system_scenarios.txt')
    initial_user_scenarios = file_to_string(f'{prompt_dir}/initial_user_scenarios.txt')
    initial_system_rewards = file_to_string(f'{prompt_dir}/initial_system_rewards.txt')
    initial_user_rewards = file_to_string(f'{prompt_dir}/initial_user_rewards.txt')

    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')

    example_scenarios = file_to_string(f'{prompt_dir}/example_scenarios.txt')
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

    example_rewards = file_to_string(f'{prompt_dir}/example_rewards.txt')
    example_rewards = example_rewards.format(
        reward_wrapper=reward_wrapper_example
    )
    example_of_o = file_to_string(f'{prompt_dir}/example_of_o.txt')
    example_of_o = example_of_o.format(obs_o=obs_o)

    code_output_tip_scenarios = file_to_string(f'{prompt_dir}/code_output_tip_scenarios.txt')
    code_output_tip_rewards = file_to_string(f'{prompt_dir}/code_output_tip_rewards.txt')
    rule_setting = file_to_string(f'{prompt_dir}/rule_setting.txt')

    main_task = "learn to play a 5 vs 5 football game"
    num_agents = 5
    num_groups = 2

    initial_system_get_decomposition = initial_system_get_decomposition.format(rule_setting=rule_setting)
    initial_user_get_decomposition = initial_user_get_decomposition.format(main_task=main_task, num_agents=num_agents,
                                                                           num_groups=num_groups)

    messages = [{"role": "system", "content": initial_system_get_decomposition},
                {"role": "user", "content": initial_user_get_decomposition}]

    DUMMY_FAILURE = -10000.
    max_scores = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None

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

    # n_dec 代表分解几层，本py用于单层深度分解 
    if n_decomposition == 1:
        logging.info(f"Decompositions Generation: GPT Output:\n " + responses[0].message.content + "\n")

    # Logging Token Information
    logging.info(
        f"Decompositions Generation: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")


    """修改plan: response 0 和 1 的循环，添加一个check，如果是0，就调用ia2c训练并保存buffer/ckpt/yaml信息
        如果是1，并且 is doe true，那么利用子团队yaml信息创建doe的yaml，调用进行训练
        问题是，这样可能需要early stop或者某种metric，不知道是reward不好还是doe的不好（需要一种缺保doe是expert的判断条件）"""



    for response_id in range(n_decomposition):

        response_cur = responses[response_id].message.content
        # responses是 len=2 的list，每个都是dict

        # 这里的命名文件等待zihao更新，用于创建env和reward
        # response id 代表分解第几种分解方案，samples；layer0代表只分解一层
        with open(f"{OUTPUT_DIR}/decomposition_layer0_decomposition{response_id}.py", 'w') as file:
            file.writelines(response_cur + '\n')

        # logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

        # Regular expression pattern
        pattern = r"\*\*Group (\d+):\*\*\n\*\*Number of agents:\*\* (\d+)\n\*\*Training goal:\*\* (.+)"

        # Search for the pattern
        matches = re.findall(pattern, response_cur)

        # Extract subtask info
        groups = []
        for match in matches:
            group_info = {
                "group_number": int(match[0]),
                "number_of_agents": int(match[1]),
                "training_goal": match[2].strip()
            }
            groups.append(group_info)

        for group in groups:
            print(f"Group {group['group_number']}:")
            print(f"Number of agents: {group['number_of_agents']}")
            print(f"Training goal: {group['training_goal']}")


        # 分别训练两个子团队任务
        # if use_doe, 每个子团队任务训练结束后的buffer要保存，用于训练 doe classifier
        for group_id in range(len(groups)):
            # Scenario generation
            logging.info(
                f"Scenarios Generation: Generating 1 sample for Decomposition {response_id} Group{group_id} with {model}")
            group = groups[group_id]
            cur_initial_system_scenarios = initial_system_scenarios.format(
                main_task_scenario=five_vs_five_code_string) + code_output_tip_scenarios + example_scenarios
            cur_initial_user_scenarios = initial_user_scenarios.format(training_goal=group['training_goal'],
                                                                       number_of_agents=group['number_of_agents'],
                                                                       scenario_builder_code_string=scenario_builder_code_string)
            cur_messages_s = copy.deepcopy(messages)
            cur_messages_s.append({"role": "assistant", "content": response_cur})
            cur_messages_s.append({"role": "system", "content": cur_initial_system_scenarios})
            cur_messages_s.append({"role": "user", "content": cur_initial_user_scenarios})

            response_scenario_cur = client.chat.completions.create(model=model,
                                                                   messages=cur_messages_s,
                                                                   temperature=temperature,
                                                                   n=1)
            reply_scenario = response_scenario_cur.choices[0].message.content
            # 提取prompt和env代码，生成训练任务scenario

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
                    scenario_code_string = scenario_code_string.group(1).strip()
                    break
            scenario_code_string = reply_scenario if not scenario_code_string else scenario_code_string

            print("Scenario Code String 1:", scenario_code_string)

            # Remove unnecessary imports
            lines = scenario_code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    scenario_code_string = "\n".join(lines[i:])

            print("Scenario Code String 2:", scenario_code_string)

            # Save the new environment code when the output contains valid code string!
            with open(f"{MAP_DIR}/scenario_layer0_decomposition{response_id}_subtask{group_id}.py", 'w') as file:
                file.writelines("from . import *" + '\n')
                file.writelines(scenario_code_string + '\n')

            with open(f"{OUTPUT_DIR}/scenario_layer0_decomposition{response_id}_subtask{group_id}.py", 'w') as file:
                file.writelines("from . import *" + '\n')
                file.writelines(scenario_code_string + '\n')

            # Save the scenario in the GRF Env
            with open(f"{GFOOTBALL_DIR}/scenarios/{Time}/scenario_layer0_decomposition{response_id}_subtask{group_id}.py",
                      'w') as file:
                file.writelines("from . import *" + '\n')
                file.writelines(scenario_code_string + '\n')

            # 生成子任务的环境代码保存到py文件

            # Reward generation and improving:
            logging.info(
                f"Rewards Generation: Generating {n_reward} samples for Decomposition {response_id} Group{group_id} with {model}")

            curr_code_output_tip_rewards = code_output_tip_rewards.format(number_of_agents=group['number_of_agents'],
                                                                          example_of_o=example_of_o)
            cur_initial_system_rewards = initial_system_rewards + example_rewards + curr_code_output_tip_rewards
            cur_initial_user_rewards = initial_user_rewards.format(training_goal=group['training_goal'],
                                                                   number_of_agents=group['number_of_agents'],
                                                                   env_code=env_code, wrappers=wrappers)

            cur_messages_r = copy.deepcopy(messages)
            cur_messages_r.append({"role": "assistant", "content": response_cur})
            # cur_messages_r.append({"role": "assistant", "content": reply_scenario})
            cur_messages_r.append({"role": "system", "content": cur_initial_system_rewards})
            cur_messages_r.append({"role": "user", "content": cur_initial_user_rewards})


            # 尝试几次 reward 生成 batch，默认2
            for i in range(n_improve_iter):
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

                    # response id是分解几层，response r id是sample的reward function个数
                    with open(
                            f"{REWARD_DIR}/reward_layer0_decomposition{response_id}_subtask{group_id}_sample{response_r_id}.py",
                            'w') as file:
                        file.writelines(reward_code_string + '\n')

                    with open(
                            f"{OUTPUT_DIR}/reward_layer0_decomposition{response_id}_subtask{group_id}_sample{response_r_id}.py",
                            'w') as file:
                        file.writelines(reward_code_string + '\n')

                    # Save the reward function in the GRF Env
                    with open(
                            f"{GFOOTBALL_DIR}/rewards/reward_layer0_decomposition{response_id}_subtask{group_id}_sample{response_r_id}.py",
                            'w') as file:
                        file.writelines(reward_code_string + '\n')

                    # Create Task YAML files
                    create_task(CONFIG_ROOT_DIR, task, 0, response_id, response_r_id, group['number_of_agents'],
                                group['group_number'] - 1, suffix)
                    
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

                    # Execute the python file with flags
                    rl_filepath = f"env_decomposition{response_id}_subtask{group_id}.txt"
                    with open(rl_filepath, 'w') as f:
                        script_path = f'{SRC_DIR}/main.py'
                        params = [
                            'python', '-u', script_path,
                            f'--config={alg_cfg}',
                            f'--env-config={task}{suffix}_decomposition{response_id}_subtask{group_id}',
                        ]
                        process = subprocess.Popen(params, stdout=f, stderr=f)
                    # Modified the check of successful training
                    block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                    rl_runs.append(process)

                # 完成了reward次数的RL training，收集了所有的traj

                # Gather RL training results and construct reward reflection
                code_feedbacks = []
                contents = []
                # May change to winning rate
                score_reward_mean = []
                code_paths = []

                exec_success = False
                for response_r_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
                    rl_run.communicate()
                    rl_filepath = f"env_layer0_decomposition{response_id}_subtask{group_id}_sample{response_r_id}.txt"
                    code_paths.append(
                        f"reward_layer0_decomposition{response_id}_subtask{group_id}_sample{response_r_id}.py")
                    try:
                        with open(rl_filepath, 'r') as f:
                            stdout_str = f.read()
                    except:
                        # content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                        content = execution_error_feedback.format(
                            traceback_msg="Code Run cannot be executed because reward class is wrongly formatted! Please re-write an entirely new reward function!")
                        content += code_output_tip_rewards
                        contents.append(content)
                        score_reward_mean.append(DUMMY_FAILURE)
                        continue

                    content = ''
                    traceback_msg = filter_traceback(stdout_str)

                    if traceback_msg == '':
                        # If RL execution has no error, provide policy statistics feedback
                        exec_success = True
                        lines = stdout_str.split('\n')

                        # Modify need: Check the result and change the epoch_freq here
                        for i, line in enumerate(lines):
                            if line.startswith('Tensorboard Directory:'):
                                break
                        tensorboard_logdir = line.split(':')[-1].strip()
                        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                        max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                        epoch_freq = max(int(max_iterations // 10), 1)

                        content += policy_feedback.format(epoch_freq=epoch_freq)

                        # Modify need: Check the result reward format
                        # Add reward components log to the feedback
                        for metric in tensorboard_logs:
                            if "/" not in metric:
                                metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                                if "score_reward_mean" == metric:
                                    score_reward_mean.append(tensorboard_logs[metric])
                                    content += f"score_reward_mean: {metric_cur}\n"

                        code_feedbacks.append(code_feedback)
                        content += code_feedback
                    else:
                        # Otherwise, provide execution traceback error feedback
                        score_reward_mean.append(DUMMY_FAILURE)
                        content += execution_error_feedback.format(traceback_msg=traceback_msg)

                    content += code_output_tip_rewards
                    contents.append(content)

                # Repeat the iteration if all code generation failed
                if not exec_success and n_decomposition != 1:
                    execute_rates.append(0.)
                    max_scores.append(DUMMY_FAILURE)
                    max_successes_reward_correlation.append(DUMMY_FAILURE)
                    best_code_paths.append(None)
                    logging.info(
                        "All code generation failed! Repeat this iteration from the current message checkpoint!")
                    continue

                # Select the best code sample based on the success rate
                best_sample_idx = np.argmax(np.array(score_reward_mean))
                best_content = contents[best_sample_idx]

                max_score = score_reward_mean[best_sample_idx]
                # max_success_reward_correlation = reward_correlations[best_sample_idx]
                execute_rate = np.sum(np.array(score_reward_mean) >= 0.) / n_decomposition

                # Update the best Eureka Output
                if max_score > max_score_overall:
                    max_score_overall = max_score
                    max_reward_code_path = code_paths[best_sample_idx]

                execute_rates.append(execute_rate)
                max_scores.append(max_score)
                best_code_paths.append(code_paths[best_sample_idx])

                logging.info(f"Iteration {iter}: Max Score: {max_score}, Execute Rate: {execute_rate}")
                logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
                logging.info(f"Iteration {iter}: GPT Output Content:\n" + responses[best_sample_idx]["message"][
                    "content"] + "\n")
                logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

                if len(messages) == 2:
                    cur_messages_r += [
                        {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
                    cur_messages_r += [{"role": "user", "content": best_content}]
                else:
                    assert len(messages) == 4
                    cur_messages_r[-2] = {"role": "assistant",
                                          "content": responses[best_sample_idx]["message"]["content"]}
                    cur_messages_r[-1] = {"role": "user", "content": best_content}

            if max_reward_code_path is None:
                logging.info("All iterations of code generation failed, aborting...")
                logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
                exit()

            # 完成了第一个子任务 reward 生成

        # 完成了 所有子任务 reward 生成，开始train merge team
        """ 
        上面第一阶段子任务训练 alg_cfg 需要用 ia2c/ia2c_ns，不能用 doe 干扰RL训练
        但是需要 save buffer 并得到 doe cls，然后在第二阶段 merge train 时 merge cls
        这种写法是每个子任务自己的性能提升自己的表现，先用着，以后的研究中再考虑与merge后技能的表现
        """

        """To LZH
        这里暂时采用的绝对路径 buffer_dir 其实就是一组分解plan中每个阶段的doe相关数据，
        比如分解方案一，分解两层，
        路径就是 results/buffer/grf/plan1 + layer0_group0_buffer.pt & layer0_group0_doe.pt & layer0_group1_doe.pt etc.
        需要调整对应的命名方式，上面只是举例，

        以及这里 decompose_id = 0 是按照第0个decompose方案的意思设计的，如果不对可以改，主要影响 yaml 文件命名
        """
        train_merge_team(groups, use_doe, decompose_id=0, buffer_dir='GRF_SUBTASK/doe_epymarl-main/results/buffer/grf')
        
    # 完成了所有方案 n decomposition plan 的任务生成，Execute the Main task using w/w. DOE:



if __name__ == "__main__":
    main(model="gpt-3.5-turbo", n_decomposition=1, n_reward=1, temperature=1, task="gfootball", alg_cfg="ia2c",
         use_doe=True, n_improve_iter=2)