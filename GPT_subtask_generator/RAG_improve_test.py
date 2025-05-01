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

OpenAI.api_base = "https://api.ohmygpt.com"
client = OpenAI(api_key="key")
logging.basicConfig(level=logging.INFO)

eval_reward = """
import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_bonus = 0.05
        self.dribble_completion_bonus = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward),
                      "dribble_completion_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Ensure length of reward and observation matches
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            actions = o['sticky_actions']

            # Check if pass action was successful
            if actions[0] == 0 and self.sticky_actions_counter[0] == 1:  # Pass action index
                components["pass_completion_reward"][rew_index] = self.pass_completion_bonus
                reward[rew_index] += self.pass_completion_bonus

            # Check if dribble action was successful
            if actions[9] == 0 and self.sticky_actions_counter[9] == 1:  # Dribble action index
                components["dribble_completion_reward"][rew_index] = self.dribble_completion_bonus
                reward[rew_index] += self.dribble_completion_bonus

            # Update sticky actions
            self.sticky_actions_counter = actions.copy()

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)

        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info

        """

eval_training_goal = "Concentrate on skills that aid in the transition from defense to attack such as Short Pass, Long Pass, and Dribble, ensuring control and movement of the ball under pressure."
eval_num_agents = 3
task = "gfootball"


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

ROOT_DIR = os.getcwd()
prompt_dir = f'{ROOT_DIR}/utils/prompts'

with open('RAG_data/database/top_10_results.json', 'r', encoding='utf-8') as file:
    top10_data = json.load(file)

top10_input = ""

# 遍历 top10 数据并格式化成适合 GPT 的输入
for i, entry in enumerate(top10_data):
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


initial_system = file_to_string(f'{prompt_dir}/RAG/initial_system_improve.txt')
initial_user = file_to_string(f'{prompt_dir}/RAG/initial_user_improve.txt')
reward_wrapper_example = f'{ROOT_DIR}/env_code/{task}/reward_wrapper_example.py'
obs_o = f'{ROOT_DIR}/env_code/{task}/obs_o.py'
obs_exp = f'{ROOT_DIR}/env_code/{task}/obs_exp.py'
reward_wrapper_example = file_to_string(reward_wrapper_example)
obs_o = file_to_string(obs_o)
obs_exp = file_to_string(obs_exp)

example_rewards = file_to_string(f'{prompt_dir}/{task}/example_rewards.txt')
example_rewards = example_rewards.format(
    reward_wrapper=reward_wrapper_example
)
example_of_o = file_to_string(f'{prompt_dir}/{task}/example_of_o.txt')
example_of_o = example_of_o.format(obs_o=obs_o, obs_exp=obs_exp)
reward_signature = file_to_string(f'{prompt_dir}/{task}/reward_signature')
code_output_tip_rewards = file_to_string(f'{prompt_dir}/{task}/code_output_tip_rewards.txt')
curr_code_output_tip_rewards = code_output_tip_rewards.format(number_of_agents=eval_num_agents,
                                                              example_of_o=example_of_o,
                                                              reward_signature=reward_signature)
env = f'{ROOT_DIR}/env_code/{task}/football_env.py'
env_core = f'{ROOT_DIR}/env_code/{task}/football_env_core.py'
observation_processor = f'{ROOT_DIR}/env_code/{task}/observation_processor.py'
env_code_string = file_to_string(env)
env_core_code_string = file_to_string(env_core)
observation_processor_code_string = file_to_string(observation_processor)

env_code = env_code_string + env_core_code_string + observation_processor_code_string

cur_initial_system = initial_system + example_rewards + curr_code_output_tip_rewards

cur_initial_user = initial_user.format(training_goal=eval_training_goal,
                                       num_agents= eval_num_agents,
                                       reward_function=eval_reward,
                                       top_10= top10_input,
                                       env_code = env_code)
print(cur_initial_user)
messages_evaluate = [{"role": "system", "content": cur_initial_system},
                     {"role": "user", "content": cur_initial_user}]
print(messages_evaluate)

response_cur = None
attempt = 0
max_attempts = 10  # 限制最大尝试次数
chunk_size = 1  # 只生成 1 个回复

while attempt < max_attempts:
    print("ATTEMPT:", attempt)
    try:
        response_cur = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages_evaluate,
            temperature=1,
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

# Regex patterns to extract python code enclosed in GPT response
patterns = [
    r'```python(.*?)```',
    r'```(.*?)```',
    r'"""(.*?)"""',
    r'""(.*?)""',
    r'"(.*?)"',
]
for pattern in patterns:
    reward_code_string = re.search(pattern, response, re.DOTALL)
    if reward_code_string is not None:
        reward_code_string = reward_code_string.group(1).strip()
        break
reward_code_string = response  if not reward_code_string else reward_code_string

print("Reward Code String 1:", reward_code_string)

# Remove unnecessary imports
lines = reward_code_string.split("\n")
for i, line in enumerate(lines):
    if line.strip().startswith("class "):
        reward_code_string = "\n".join(lines[i:])

print("Reward Code String 2:", reward_code_string)

# response id是分解几层，response r id是sample的reward function个数
with open("reward_new.py", 'w') as file:
    file.writelines("import gym" + '\n')
    file.writelines("import numpy as np" + '\n')
    file.writelines(reward_code_string + '\n')

