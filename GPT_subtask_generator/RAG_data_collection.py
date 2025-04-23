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
from utils.create_task import create_task, create_train_cfg,create_task_RAG, create_train_cfg_RAG
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
client = OpenAI(api_key="KEY")

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
REWARD_DATA_DIR = f"RAG_data/reward_functions/{Time}"
TRAIN_LOG_DIR = f"RAG_data/training_log/{Time}"
TRAJECTORY_DIR = f"RAG_data/trajectory/{Time}"
MAP_DIR = f"{MAP_DIR}/{Time}"
REWARD_DIR = f"{REWARD_DIR}/{Time}"
SACRED_DIR = os.path.join(MEDoE_DIR, "doe_epymarl-main/results/sacred")
TENSORBOARD_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/results/tb_logs')
# GRF_SCENARIO_DIR = f"{GFOOTBALL_DIR}/scenarios/{Time}"

# 创建目标文件夹（如果不存在）
os.makedirs(REWARD_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
os.makedirs(TRAJECTORY_DIR, exist_ok=True)
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




def main(model, n_reward, map_name, temperature, task, alg_cfg):
    num_agents =5
    training_goal = "Focused on developing offensive strategies including mastering accurate shooting, effective dribbling to evade opponents, and practicing different pass types (long and high passes) to break defensive lines."

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    suffix = "_GPT"
    logging.info(f"Using LLM: {model}")

    # env_init = f'{ROOT_DIR}/env_code/__init__.py'
    env = f'{ROOT_DIR}/env_code/{task}/football_env.py'
    env_core = f'{ROOT_DIR}/env_code/{task}/football_env_core.py'
    # action_set= f'{ROOT_DIR}/env_code/{task}/football_action_set.py'
    observation_processor = f'{ROOT_DIR}/env_code/{task}/observation_processor.py'
    scenario_builder = f'{ROOT_DIR}/env_code/{task}/scenario_builder.py'
    reward_wrapper_example = f'{ROOT_DIR}/env_code/{task}/reward_wrapper_example.py'
    obs_o = f'{ROOT_DIR}/env_code/{task}/obs_o.py'
    obs_exp = f'{ROOT_DIR}/env_code/{task}/obs_exp.py'

    execution_error_feedback = file_to_string(f'{prompt_dir}/{task}/execution_error_feedback.txt')


    env_code_string = file_to_string(env)
    env_core_code_string = file_to_string(env_core)
    observation_processor_code_string = file_to_string(observation_processor)

    env_code = env_code_string + env_core_code_string + observation_processor_code_string

    reward_wrapper_example = file_to_string(reward_wrapper_example)
    obs_o = file_to_string(obs_o)
    obs_exp = file_to_string(obs_exp)


    # Loading all text prompts
    initial_system = file_to_string(f'{prompt_dir}/RAG/initial_system.txt')
    initial_user = file_to_string(f'{prompt_dir}/RAG/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/{task}/reward_signature')

    # execution_error_feedback = file_to_string(f'{prompt_dir}/{task}/execution_error_feedback.txt')
    # code_feedback = file_to_string(f'{prompt_dir}/{task}/code_feedback.txt')
    # policy_feedback = file_to_string(f'{prompt_dir}/{task}/policy_feedback.txt')

    example_rewards = file_to_string(f'{prompt_dir}/{task}/example_rewards.txt')
    example_rewards = example_rewards.format(
        reward_wrapper=reward_wrapper_example
    )
    example_of_o = file_to_string(f'{prompt_dir}/{task}/example_of_o.txt')
    example_of_o = example_of_o.format(obs_o=obs_o, obs_exp=obs_exp)


    code_output_tip_rewards = file_to_string(f'{prompt_dir}/{task}/code_output_tip_rewards.txt')
    rule_setting = file_to_string(f'{prompt_dir}/{task}/rule_setting.txt')

    curr_code_output_tip_rewards = code_output_tip_rewards.format(number_of_agents= num_agents,
                                                                  example_of_o=example_of_o,
                                                                  reward_signature=reward_signature)
    cur_initial_system = initial_system + example_rewards + curr_code_output_tip_rewards
    cur_initial_user = initial_user.format(training_goal= training_goal,
                                           env_code=env_code, )

    messages = [{"role": "system", "content": cur_initial_system},
                {"role": "user", "content": cur_initial_user}]

    #######################################################################################################################
    # Get response
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    # chunk_size = sample if "gpt-3.5" in model else 4
    chunk_size = 5

    while True:
        if total_samples >= n_reward:
            break
        for attempt in range(1000):
            print("ATTEMPT:", attempt)
            try:
                response_cur = client.chat.completions.create(model=model,
                                                              messages=messages,
                                                              temperature=temperature,
                                                              n=chunk_size)
                total_samples += chunk_size
                logging.info(f"Number of generated rewards: {total_samples} ")
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


    # Logging Token Information
    logging.info(
        f"Reward Generation: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

    # Save the reward functions
    for response_id in range(n_reward):

        reply_reward = responses[response_id].message.content
        # responses是 len=2 的list，每个都是dict
        logging.info(f"Saving reward: {response_id}")
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
                f"{REWARD_DIR}/reward_{response_id}.py",
                'w') as file:
            file.writelines("import gym" + '\n')
            file.writelines("import numpy as np" + '\n')
            file.writelines(reward_code_string + '\n')

        with open(
                f"{REWARD_DATA_DIR}/reward_{response_id}.py",
                'w') as file:
            file.writelines("import gym" + '\n')
            file.writelines("import numpy as np" + '\n')
            file.writelines(reward_code_string + '\n')

        # Save the reward function in the GRF Env
        with open(
                f"{GFOOTBALL_DIR}/rewards/reward_{response_id}.py",
                'w') as file:
            file.writelines("import gym" + '\n')
            file.writelines("import numpy as np" + '\n')
            file.writelines(reward_code_string + '\n')

    #######################################################################################################################
    #Train using reward functions
    for response_id in range(n_reward):
        logging.info(f"Training using reward: {response_id}")
        # Create Task YAML files
        create_task_RAG(CONFIG_ENVS_DIR, task, num_agents,response_id, map_name)
        create_train_cfg_RAG(CONFIG_ALGS_DIR, Time, alg_cfg, response_id, num_agents)

        TIMEOUT = 30
        # Execute the python file with flags
        rl_filepath = f"{TRAIN_LOG_DIR}/{task}_reward_{response_id}.txt"
        with open(rl_filepath, 'w') as f:
            script_path = f'{SRC_DIR}/main.py'
            params = [
                'python', '-u', script_path,
                f'--config={alg_cfg}_RAG_reward{response_id}',
                f'--env-config={task}_RAG_reward{response_id}',
            ]
            process = subprocess.Popen(params, stdout=f, stderr=f)

            # 获取文件的初始修改时间
            while True:
                initial_mtime = os.path.getmtime(rl_filepath)
                initial_mtime = datetime.datetime.fromtimestamp(initial_mtime)  # 时间转为datetime格式
                start_time = datetime.datetime.now()
                delta_time = start_time - initial_mtime   # 时间差
                delta_seconds = delta_time.total_seconds()  # 时间差转成秒
                if delta_seconds > TIMEOUT:  # 如果文件更新时间大于30秒，重新启动程序
                    print(f"Overtime：It seems that the training is stuck or finished, subprocess terminates")
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

    #######################################################################################################################
    #Copy the training results
    # 源文件夹 A 和目标文件夹 B
    tb_folder = f"{TENSORBOARD_DIR}/{Time}"
    sacred_folder = f"{SACRED_DIR}/{Time}/{map_name}"
    target_folder = TRAJECTORY_DIR

    # 获取 A 目录下的所有文件夹，按 layer 顺序排序
    subfolders = sorted([f for f in os.listdir(tb_folder) if os.path.isdir(os.path.join(tb_folder, f))])

    for i, folder_name in enumerate(subfolders):
        # 生成新文件夹名称
        new_folder_name = f"tb_log_{i}"
        new_folder_path = os.path.join(target_folder, new_folder_name)

        # 创建目标文件夹
        os.makedirs(new_folder_path, exist_ok=True)

        # 复制源文件夹中的所有内容到新文件夹
        source_path = os.path.join(tb_folder, folder_name)
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            target_path = os.path.join(new_folder_path, item)

            if os.path.isdir(item_path):
                shutil.copytree(item_path, target_path, dirs_exist_ok=True)  # 复制文件夹
            else:
                shutil.copy2(item_path, target_path)  # 复制文件

    for item in os.listdir(sacred_folder):
        src_path = os.path.join(sacred_folder, item)
        dst_path = os.path.join(target_folder, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)  # 复制整个文件夹
        else:
            shutil.copy2(src_path, dst_path)  # 复制文件

# #######################################################################################################################
#     # Evaluate using the trajectory
#     for response_id in range(n_reward):
#         exec_success = False
#         reward_cur = file_to_string(f'{REWARD_DATA_DIR}/reward_{response_id}.py')
#         # try:
#         #     with open(rl_filepath, 'r') as f:
#         #         stdout_str = f.read()
#         # except:
#         #     content = ("Code Run cannot be executed because reward class is wrongly formatted!")
#         #     Error = True
#         #     continue
#         with open(f"{TRAIN_LOG_DIR}/{task}_reward_{response_id}.txt", 'r') as f:
#             stdout_str = f.read()
#
#         content = ''
#         traceback_msg = filter_traceback(stdout_str)
#
#         if traceback_msg == '':
#             print("No errors in the reward function")
#             # If RL execution has no error, provide policy statistics feedback
#             exec_success = True
#             exec_message = f"This reward function can be successfully executed."
#
#             with open(f"{TRAJECTORY_DIR}/scoring,reward_{response_id}/1/metrics.json", "r") as file:
#                 data = json.load(file)
#
#             corresponding_episodes = data["episode"]['values']
#             content += f"corresponding episodes: {corresponding_episodes[::10]} \n"
#
#             component_values = {
#                 key: value['values'] for key, value in data.items() if key.startswith('component')
#             }
#             for key, value in component_values.items():
#                 value = value[::10]
#                 content += f"{key}: {value} \n"
#
#             score_reward_mean_values = data["score_reward_mean"]['values']
#             # 获取最后10个值
#             last_10_values = score_reward_mean_values[-10:]
#             # 计算平均数
#             average = sum(last_10_values) / len(last_10_values)
#
#             content += f"score_reward_mean: {score_reward_mean_values[::10]} \n"
#
#             final_reward_mean_values = data["final_reward_mean"]['values']
#             content += f"final_reward_mean: {final_reward_mean_values[::10]} \n"
#
#         else:
#             print("Spotting errors in the reward function")
#             # Otherwise, provide execution traceback error feedback
#             exec_success = False
#             exec_message = f"This reward function can NOT be successfully executed."
#             content += execution_error_feedback.format(traceback_msg=traceback_msg)
#
#         # initial_user_evaluate和initial_system_evaluate需要修改








if __name__ == "__main__":
    main(model="gpt-4-turbo", n_reward=50, map_name="5_vs_5", temperature=1, task="gfootball", alg_cfg="ia2c")