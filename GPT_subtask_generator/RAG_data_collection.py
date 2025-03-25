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
import multiprocessing

current_dir = os.path.dirname(os.path.abspath(__file__))
MEDoE_DIR = os.path.abspath(os.path.join(current_dir, "../"))

SRC_DIR = os.path.join(MEDoE_DIR, "doe_epymarl-main/src")
sys.path.append(SRC_DIR)

from run import *

OpenAI.api_base = "https://api.ohmygpt.com"
client = OpenAI(api_key="KEY")
logging.basicConfig(level=logging.INFO)

# ROOT_DIR = os.getcwd()
# parent_dir = os.path.dirname(ROOT_DIR)
# GFOOTBALL_DIR = os.path.join(MEDoE_DIR, "gfootball")
# CONFIG_ENVS_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/config/envs')
# CONFIG_ALGS_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/config/algs')
# MAP_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/maps')
# REWARD_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/rewards')
# prompt_dir = f'{ROOT_DIR}/utils/prompts'
# logging.basicConfig(level=logging.INFO)

# Time = datetime.datetime.now()
# Time = Time.strftime("%Y-%m-%d-%H-%M-%S")
# Time = "2025-03-09-22-26-00"
# REWARD_DATA_DIR = f"RAG_data/reward_functions/{Time}"
# TRAIN_LOG_DIR = f"RAG_data/training_log/{Time}"
# TRAJECTORY_DIR = f"RAG_data/trajectory/{Time}"
# MAP_DIR = f"{MAP_DIR}/{Time}"
# REWARD_DIR = f"{REWARD_DIR}/{Time}"
# SACRED_DIR = os.path.join(MEDoE_DIR, "doe_epymarl-main/results/sacred")
# TENSORBOARD_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/results/tb_logs')
# # GRF_SCENARIO_DIR = f"{GFOOTBALL_DIR}/scenarios/{Time}"
#
# # 创建目标文件夹（如果不存在）
# os.makedirs(REWARD_DATA_DIR, exist_ok=True)
# os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
# os.makedirs(TRAJECTORY_DIR, exist_ok=True)
# os.makedirs(MAP_DIR, exist_ok=True)
# os.makedirs(REWARD_DIR, exist_ok=True)
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


# 并行训练的函数
def train_reward(response_id, task, num_agents, map_name, alg_cfg, gpu_id, CONFIG_ENVS_DIR, CONFIG_ALGS_DIR, TRAIN_LOG_DIR, Time):
    logging.info(f"Training using reward: {response_id} on GPU {gpu_id}")

    # 设置CUDA_VISIBLE_DEVICES为当前的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Create Task YAML files
    create_task_RAG(CONFIG_ENVS_DIR, task, num_agents, response_id, map_name)
    create_train_cfg_RAG(CONFIG_ALGS_DIR, Time, alg_cfg, response_id, num_agents)

    TIMEOUT = 30
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
            initial_mtime = datetime.datetime.fromtimestamp(initial_mtime)
            start_time = datetime.datetime.now()
            delta_time = start_time - initial_mtime
            delta_seconds = delta_time.total_seconds()

            if delta_seconds > TIMEOUT:
                print(f"Overtime: It seems that the training is stuck or finished, subprocess terminates")
                process.kill()
                break

            time.sleep(1)

        process.wait()


# 复制文件的函数
def copy_files(source_path, target_folder, i):
    new_folder_name = f"tb_log_{i}"
    new_folder_path = os.path.join(target_folder, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        target_path = os.path.join(new_folder_path, item)

        if os.path.isdir(item_path):
            shutil.copytree(item_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(item_path, target_path)


# 复制整个文件夹的函数
def copy_sacred_folder(src_folder, target_folder):
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(target_folder, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

tasks = [
    {"num_agents": 4, "training_goal": "Focused on developing offensive strategies including mastering accurate shooting, effective dribbling to evade opponents, and practicing different pass types (long and high passes) to break defensive lines."},
    {"num_agents": 2, "training_goal": "Focus on mastering offensive maneuvers by enhancing specialized quick attack responses and dynamic adaptation during varied game phases."},
    {"num_agents": 3, "training_goal": "Enhance teamwork and coordination in defensive strategies, focusing on seamless transitions between maintaining ball control and executing strategic defensive positioning."},
    {"num_agents": 2, "training_goal": "Master the skill of shooting, focusing on accuracy and power, to effectively capitalize on scoring opportunities against the opposition's defense. Emphasis on practicing 'Shot' with different pressure scenarios during game simulations."},
    {"num_agents": 2, "training_goal": "Develop advanced dribbling techniques, including evasion and ball control, paired with 'Sprint' to increase speed and agility in offensive positions. Train in scenarios that mimic breaking through tight defensive lines and evading defenders."},
    {"num_agents": 2, "training_goal": "Refine defensive skills by specializing training in immediate tactical responses and developing quick transition skills for counter-attacks."},
    {"num_agents": 2, "training_goal": "Specialize in learning energy conservation techniques through proficient usage of Stop-Sprint and Stop-Moving, crucial for maintaining stamina and positional integrity over the duration of a match."},
    {"num_agents": 2, "training_goal": "Master defensive maneuvers specifically sliding tackles, focusing intensely on the timing and precision of these moves under high-pressure situations."},
    {"num_agents": 2, "training_goal": "Train to excel in quick decision-making and efficient ball handling to initiate counter-attacks immediately after recovering possession."},
    {"num_agents": 3, "training_goal": "Learn offensive skills such as passing, shooting, and dribbling to create scoring opportunities. Specifically, focus on actions like Short Pass, Long Pass, Shot, Dribble, and Sprint."},
    {"num_agents": 2, "training_goal": "Focus on defensive skills such as positioning, interception, marking, and tackling to prevent the opponent from scoring. Specifically, actions to focus on include Sliding, Stop-Dribble, and Stop-Moving to effectively block the opponents' attacks."},
    {"num_agents": 2, "training_goal": "Focus on enhancing offensive capabilities through specialized training focused on fast-paced maneuvers and precision-based finishing control."},
    {"num_agents": 2, "training_goal": "This agent functions as a hybrid of midfielder/advance defender, trained to excel in both offensive and defensive transitions. The agent should be adept in High Pass and Long Pass to switch play and assist in build-ups, as well as Dribble under pressure to maintain possession and provide time for other players to position. This agent will also learn Sprint and Stop Sprint to effectively respond to the changing dynamics of the game."},
    {"num_agents": 2, "training_goal": "Focus on the role of a 'stopper,' enhancing skills in intense man-marking, blocking shots, and stalling forward moves by opposing players."},
    {"num_agents": 2,"training_goal": "Focus on the role of a 'sweeper,' adept at clearing the ball from the defensive zone, performing critical last-man tackles, and supporting the stopper by covering positions and executing fast recoveries."},
    {"num_agents": 2, "training_goal": "Concentrate on mastering the technical aspects and precision of long passes. Training includes understanding the dynamics of ball travel over different lengths and practicing accuracy under varying match conditions."},
    {"num_agents": 2, "training_goal": "Focus on the technical skill enhancement necessary for executing high passes with precision. This includes trajectory control, power assessment, and situational application drills focusing on scenarios where high passes are advantageous."},
    {"num_agents": 2, "training_goal": "Concentrate on wide midfield responsibilities, mastering High Pass and positioning to expand the field of play and support lateral transitions, aiding in stretching the opposition's defense and creating space."},
    {"num_agents": 3, "training_goal": "Enhance the synergistic effectiveness of the central midfield by focusing on seamless transitions and controlled pace management, ensuring that both tactical elements support one another for more robust gameplay."},
    {"num_agents": 7, "training_goal": "Develop specialized groups focusing on precise defense and strategic midfield management, with emphasis on interplay between defensive maneuvers and midfield control for optimized transitions and ball handling."},
    {"num_agents": 5, "training_goal": "Focus on offensive strategies, optimizing team coordination and reaction to force openings and defense breaking, adapting seamlessly between immediate scoring strategies (shooting) and long-term placement strategies (passing and positioning)."},
    {"num_agents": 4, "training_goal": "Focus on mastering defensive responsiveness and interception skills, with specialized training tailored to high-pressure defensive scenarios and general defensive positioning."},
    {"num_agents": 2, "training_goal": "Focus on Sprint techniques to improve general defensive coverage, enabling quicker positioning across the field to adapt to changing game dynamics."},
    {"num_agents": 2, "training_goal": "Focus on synergizing defensive roles between two agents in high-pressure defensive scenarios to cover broader tactics, enhancing coordination and specific role proficiency near the penalty area."},
]


def main(model, n_reward, map_name, temperature, task, alg_cfg, num_agents, training_goal, task_index):
    ROOT_DIR = os.getcwd()
    parent_dir = os.path.dirname(ROOT_DIR)
    GFOOTBALL_DIR = os.path.join(MEDoE_DIR, "gfootball")
    CONFIG_ENVS_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/config/envs')
    CONFIG_ALGS_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/config/algs')
    MAP_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/maps')
    REWARD_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/src/envs/gfootball/rewards')
    prompt_dir = f'{ROOT_DIR}/utils/prompts'
    # Time = datetime.datetime.now()
    # Time = Time.strftime("%Y-%m-%d-%H-%M-%S")
    Time = f"rag_task_{task_index}"
    REWARD_DATA_DIR = f"RAG_data/reward_functions/{Time}"
    TRAIN_LOG_DIR = f"RAG_data/training_log/{Time}"
    TRAJECTORY_DIR = f"RAG_data/trajectory/{Time}"
    DATABASE_DIR = f"RAG_data/database/{Time}"
    MAP_DIR = f"{MAP_DIR}/{Time}"
    REWARD_DIR = f"{REWARD_DIR}/{Time}"
    SACRED_DIR = os.path.join(MEDoE_DIR, "doe_epymarl-main/results/sacred")
    TENSORBOARD_DIR = os.path.join(MEDoE_DIR, 'doe_epymarl-main/results/tb_logs')
    # GRF_SCENARIO_DIR = f"{GFOOTBALL_DIR}/scenarios/{Time}"

    # 创建目标文件夹（如果不存在）
    os.makedirs(REWARD_DATA_DIR, exist_ok=True)
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(MAP_DIR, exist_ok=True)
    os.makedirs(REWARD_DIR, exist_ok=True)

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
    initial_system_evaluate = file_to_string(f'{prompt_dir}/RAG/initial_system_evaluate.txt')
    initial_user_evaluate = file_to_string(f'{prompt_dir}/RAG/initial_user_evaluate.txt')
    initial_user_evaluate_error = file_to_string(f'{prompt_dir}/RAG/initial_user_evaluate_error.txt')

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
    # # Get response
    # responses = []
    # response_cur = None
    # total_samples = 0
    # total_token = 0
    # total_completion_token = 0
    # # chunk_size = sample if "gpt-3.5" in model else 4
    # chunk_size = 5
    #
    # while True:
    #     if total_samples >= n_reward:
    #         break
    #     for attempt in range(1000):
    #         print("ATTEMPT:", attempt)
    #         try:
    #             response_cur = client.chat.completions.create(model=model,
    #                                                           messages=messages,
    #                                                           temperature=temperature,
    #                                                           n=chunk_size)
    #             total_samples += chunk_size
    #             logging.info(f"Number of generated rewards: {total_samples} ")
    #             break
    #         except Exception as e:
    #             if attempt >= 10:
    #                 chunk_size = max(int(chunk_size / 2), 1)
    #                 print("Current Chunk Size", chunk_size)
    #             logging.info(f"Attempt {attempt + 1} failed with error: {e}")
    #             time.sleep(1)
    #     if response_cur is None:
    #         logging.info("Code terminated due to too many failed attempts!")
    #         exit()
    #
    #     responses.extend(response_cur.choices)
    #     print("RESPONSES:", responses)
    #     prompt_tokens = response_cur.usage.prompt_tokens
    #     total_completion_token += response_cur.usage.completion_tokens
    #     total_token += response_cur.usage.total_tokens
    #
    #
    # # Logging Token Information
    # logging.info(
    #     f"Reward Generation: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
    #
    # # Save the reward functions
    # for response_id in range(n_reward):
    #
    #     reply_reward = responses[response_id].message.content
    #     # responses是 len=2 的list，每个都是dict
    #     logging.info(f"Saving reward: {response_id}")
    #     # Regex patterns to extract python code enclosed in GPT response
    #     patterns = [
    #         r'```python(.*?)```',
    #         r'```(.*?)```',
    #         r'"""(.*?)"""',
    #         r'""(.*?)""',
    #         r'"(.*?)"',
    #     ]
    #     for pattern in patterns:
    #         reward_code_string = re.search(pattern, reply_reward, re.DOTALL)
    #         if reward_code_string is not None:
    #             reward_code_string = reward_code_string.group(1).strip()
    #             break
    #     reward_code_string = reply_reward if not reward_code_string else reward_code_string
    #
    #     print("Reward Code String 1:", reward_code_string)
    #
    #     # Remove unnecessary imports
    #     lines = reward_code_string.split("\n")
    #     for i, line in enumerate(lines):
    #         if line.strip().startswith("class "):
    #             reward_code_string = "\n".join(lines[i:])
    #
    #     print("Reward Code String 2:", reward_code_string)
    #
    #
    #     # response id是分解几层，response r id是sample的reward function个数
    #     with open(
    #             f"{REWARD_DIR}/reward_{response_id}.py",
    #             'w') as file:
    #         file.writelines("import gym" + '\n')
    #         file.writelines("import numpy as np" + '\n')
    #         file.writelines(reward_code_string + '\n')
    #
    #     with open(
    #             f"{REWARD_DATA_DIR}/reward_{response_id}.py",
    #             'w') as file:
    #         file.writelines("import gym" + '\n')
    #         file.writelines("import numpy as np" + '\n')
    #         file.writelines(reward_code_string + '\n')
    #
    #     # Save the reward function in the GRF Env
    #     with open(
    #             f"{GFOOTBALL_DIR}/rewards/reward_{response_id}.py",
    #             'w') as file:
    #         file.writelines("import gym" + '\n')
    #         file.writelines("import numpy as np" + '\n')
    #         file.writelines(reward_code_string + '\n')

    ######################################################################################################################
    #Train using reward functions
    # 设置可用的 GPU 数量和并行任务的数量
    available_gpus = 1  # 这里设置为 1，因为你有一个 GPU
    parallel_task_count = 10  # 你可以设置并行的任务数量，这里设置为 2

    # 创建一个进程池，数量根据并行任务数量来设定
    pool = multiprocessing.Pool(processes=parallel_task_count)

    # # 为每个训练任务设置参数
    # reward_tasks = [(response_id, task, num_agents, map_name, alg_cfg, 0, CONFIG_ENVS_DIR, CONFIG_ALGS_DIR, TRAIN_LOG_DIR, Time)  # GPU 为 0
    #                 for response_id in range(n_reward)]  # 根据 n_reward 中的任务数量
    reward_tasks = [
        (response_id, task, num_agents, map_name, alg_cfg, 0, CONFIG_ENVS_DIR, CONFIG_ALGS_DIR, TRAIN_LOG_DIR, Time)
        for response_id in range(15, n_reward)  # 从 15 开始遍历到 n_reward-1
    ]

    # 使用 starmap 将并行训练任务发送到进程池
    pool.starmap(train_reward, reward_tasks)

    pool.close()
    pool.join()  # 等待所有任务完成
    #######################################################################################################################
    #Copy the training results
    # 源文件夹 A 和目标文件夹 B
    # 训练完成后再进行存储操作
    tb_folder = f"{TENSORBOARD_DIR}/{Time}"
    sacred_folder = f"{SACRED_DIR}/{Time}/{map_name}"
    target_folder = TRAJECTORY_DIR

    # subfolders = sorted([f for f in os.listdir(tb_folder) if os.path.isdir(os.path.join(tb_folder, f))])
    def extract_layer_number(folder_name):
        match = re.search(r'layer(\d+)', folder_name)
        return int(match.group(1)) if match else float('inf')  # 避免匹配失败

    # 获取并排序子文件夹
    subfolders = sorted(
        [f for f in os.listdir(tb_folder) if os.path.isdir(os.path.join(tb_folder, f))],
        key=extract_layer_number  # 按 layer 后的数字进行排序
    )

    # 再次使用进程池来并行处理文件复制
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.starmap(copy_files,
                 [(os.path.join(tb_folder, folder_name), target_folder, i) for i, folder_name in enumerate(subfolders)])

    copy_sacred_folder(sacred_folder, target_folder)

    pool.close()
    pool.join()  # 等待所有复制任务完成

# #######################################################################################################################
    RAG_datas = []
    # Evaluate using the trajectory
    for response_id in range(n_reward):
        logging.info(f"Evaluating reward: {response_id}")
        reward_cur = file_to_string(f'{REWARD_DATA_DIR}/reward_{response_id}.py')
        # try:
        #     with open(rl_filepath, 'r') as f:
        #         stdout_str = f.read()
        # except:
        #     content = ("Code Run cannot be executed because reward class is wrongly formatted!")
        #     Error = True
        #     continue
        with open(f"{TRAIN_LOG_DIR}/{task}_reward_{response_id}.txt", 'r') as f:
            stdout_str = f.read()

        content = ''
        traceback_msg = filter_traceback(stdout_str)

        if traceback_msg == '':
            print("No errors in the reward function")
            # If RL execution has no error, provide policy statistics feedback
            exec_success = True

            with open(f"{TRAJECTORY_DIR}/scoring, reward_{response_id}/1/metrics.json", "r") as file:
                data = json.load(file)

            corresponding_episodes = data["episode"]['values']
            content += f"checkpoint episodes: {corresponding_episodes} \n"

            component_values = {
                key: value['values'] for key, value in data.items() if key.startswith('component')
            }
            for key, value in component_values.items():
                content += f"{key}: {value} \n"

            score_reward_mean_values = data["score_reward_mean"]['values']
            content += f"score_reward_mean: {score_reward_mean_values} \n"

            final_reward_mean_values = data["final_reward_mean"]['values']
            content += f"final_reward_mean: {final_reward_mean_values} \n"

            final_reward_mean_values = data["final_reward_mean"]['values']
            content += f"final_reward_mean: {final_reward_mean_values} \n"

        else:
            print("Spotting errors in the reward function")
            # Otherwise, provide execution traceback error feedback
            exec_success = False
            content += execution_error_feedback.format(traceback_msg=traceback_msg)

        if exec_success:
            cur_initial_user_evaluate = initial_user_evaluate.format(training_goal = training_goal,
                                                                 num_agents=num_agents,
                                                                 reward_function = reward_cur,
                                                                 content = content)
        else:
            cur_initial_user_evaluate = initial_user_evaluate_error.format(training_goal = training_goal,
                                                                 num_agents=num_agents,
                                                                 reward_function = reward_cur,
                                                                 content = content)

        messages_evaluate = [{"role": "system", "content": initial_system_evaluate},
                             {"role": "user", "content": cur_initial_user_evaluate}]
        response_cur = None
        attempt = 0
        max_attempts = 10  # 限制最大尝试次数
        chunk_size = 1  # 只生成 1 个回复

        while attempt < max_attempts:
            print("ATTEMPT:", attempt)
            try:
                response_cur = client.chat.completions.create(
                    model=model,
                    messages=messages_evaluate,
                    temperature=temperature,
                    n=chunk_size
                )
                break  # 成功则跳出循环
            except Exception as e:
                logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(1)  # 等待 1 秒后重试
                attempt += 1

        # 如果所有尝试都失败，则终止
        if response_cur is None:
            logging.error("Failed to generate response after multiple attempts!")
            exit()

        # 解析返回的响应
        response = response_cur.choices[0].message.content  # 只取第一个回复
        print("Generated Response:", response)

        # 记录 token 信息
        logging.info(
            f"Tokens Used: Prompt {response_cur.usage.prompt_tokens}, Completion {response_cur.usage.completion_tokens}, Total {response_cur.usage.total_tokens}")

        # 定义正则表达式
        pattern = r"\*\*Evaluation:\*\*\s*(.*?)\s*\*\*Suggestions:\*\*\s*(.*)"

        # 进行匹配
        match = re.search(pattern, response, re.DOTALL)

        # 提取内容
        if match:
            evaluation = match.group(1).strip()  # 提取 Evaluation 部分
            suggestions = match.group(2).strip()  # 提取 Suggestions 部分

        RAG_data = {
            "training_goal": training_goal,
            "num_agents": num_agents,  # 假设每个任务需要不同的智能体数量
            "reward_function": reward_cur,
            "evaluation": evaluation,
            "suggestions": suggestions
        }

        # 将生成的数据追加到任务数据列表中
        RAG_datas.append(RAG_data)

    with open(f"{DATABASE_DIR}/RAG_data_{task_index}.json", "w") as json_file:
        json.dump(RAG_datas, json_file, indent=2)  # indent=2 使输出格式易读

    print("JSON file has been created successfully.")


if __name__ == "__main__":
    model="gpt-4-turbo"
    n_reward = 30
    # map_name="11_vs_11_easy_stochastic"
    map_name = "5_vs_5"
    temperature=1
    task="gfootball"
    alg_cfg="mappo"
    for task_index, task_config in enumerate(tasks[22:], start=22):
        num_agents = task_config["num_agents"]
        training_goal = task_config["training_goal"]
        logging.info(
            f"Running task {task_index + 1} (index {task_index}) with num_agents={num_agents} and training_goal={training_goal}")
        main(model, n_reward, map_name, temperature, task, alg_cfg, num_agents, training_goal, task_index)

    # for task_index, task_config in enumerate(tasks[:6], start=0):  # 遍历前五个任务
    #     num_agents = task_config["num_agents"]
    #     training_goal = task_config["training_goal"]
    #     logging.info(
    #         f"Running task {task_index + 1} (index {task_index}) with num_agents={num_agents} and training_goal={training_goal}"
    #     )
    #     main(model, n_reward, map_name, temperature, task, alg_cfg, num_agents, training_goal, task_index)

    # task_index = 23
    # task_config = tasks[task_index]
    #
    # num_agents = task_config["num_agents"]
    # training_goal = task_config["training_goal"]
    #
    # logging.info(
    #     f"Running task {task_index + 1} (index {task_index}) with num_agents={num_agents} and training_goal={training_goal}"
    # )
    #
    # main(model, n_reward, map_name, temperature, task, alg_cfg, num_agents, training_goal, task_index)
