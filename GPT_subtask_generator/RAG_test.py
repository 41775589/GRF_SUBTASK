#%%

import together, os
from together import Together
import json
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Paste in your Together AI API Key or load it
# TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_API_KEY = "key"
# print(TOGETHER_API_KEY)


with open('RAG_data/merged_knowledge_base.json', 'r') as file:
    rag_data = json.load(file)

print(len(rag_data))


def generate_embeddings(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """
    together_client = together.Together(api_key = TOGETHER_API_KEY)
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return np.array([x.embedding for x in outputs.data])


def generate_batch_embeddings(input_texts: List[str], model_api_string: str, batch_size: int = 32) -> np.ndarray:
    """将大的文本列表分批生成嵌入"""
    all_embeddings = []

    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}...")  # 打印当前批次
        batch_embeddings = generate_embeddings(batch, model_api_string)
        all_embeddings.append(batch_embeddings)

    # 将所有批次拼接起来
    return np.concatenate(all_embeddings, axis=0)

# We will concatenate fields in the dataset in prep for embedding

# to_embed = []

# for data in rag_data:
#     text = (
#         f"Training goal: {data.get('training_goal', '')}. "
#         f"Reward function: {data.get('reward_function', '')}. "
#         f"Component: {data.get('component', '')}."
#     )
#     to_embed.append(text.strip())
#
# # for data in rag_data:
# #     text = ''
# #     for field in ['training_goal', 'reward_function', 'component']:
# #         value = data.get(field, '')
# #         text += str(value) + ' '
# #     to_embed.append(text.strip())
# # print("to_embed: ", to_embed)
#
# # Use bge-base-en-v1.5 model to generate embeddings
# # embeddings = generate_embeddings(to_embed, 'togethercomputer/m2-bert-80M-2k-retrieval')
# embeddings = generate_batch_embeddings(to_embed, 'togethercomputer/m2-bert-80M-2k-retrieval')
#
# np.save('RAG_data/database/embeddings.npy', embeddings)
#
# print("Embedding shape: ", embeddings.shape)
# print("Embeddings: ", embeddings)

embeddings = np.load('RAG_data/knowledge_base_embeddings.npy')

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


# Generate the vector embeddings for the query
# 这是从LLM生成的奖励函数，或者子任务
# 随机选取rag_task_1/reward_10.py 内容用于测试
query = f"Training goal: {eval_training_goal}. Reward function: {eval_reward}."

# query = f"{eval_training_goal} {eval_reward}"

query_embedding = generate_embeddings([query], 'togethercomputer/m2-bert-80M-2k-retrieval')[0]


print("Query Embedding shape: ", query_embedding.shape)
print(" Query Embeddings: ", query_embedding)

similarity_scores = cosine_similarity([query_embedding], embeddings)

print("Similarity Scores shape: ", similarity_scores.shape)
print("Similarity Scores: ", similarity_scores)
# 相似度检索
# Get the indices of the highest to lowest values
indices = np.argsort(-similarity_scores)

top_10_sorted_suggestions = [rag_data[index]['suggestions'] for index in indices[0]][:10]
top_10_data = [rag_data[index] for index in indices[0][:10]]
print("TOP-TEN:", top_10_data)

with open('RAG_data/top_10_results.json', 'w', encoding='utf-8') as f:
    json.dump(top_10_data, f, ensure_ascii=False, indent=2)

# %%

# # 封装retrieval
# def retrieve(query: str, top_k: int = 5, index: np.ndarray = None) -> List[int]:
#     """
#     Retrieve the top-k most similar items from an index based on a query.
#     Args:
#         query (str): The query string to search for.
#         top_k (int, optional): The number of top similar items to retrieve. Defaults to 5.
#         index (np.ndarray, optional): The index array containing embeddings to search against. Defaults to None.
#     Returns:
#         List[int]: A list of indices corresponding to the top-k most similar items in the index.
#     """
#
#     query_embedding = generate_embeddings([query], 'togethercomputer/m2-bert-80M-2k-retrieval')[0]
#     similarity_scores = cosine_similarity([query_embedding], index)
#
#     return np.argsort(-similarity_scores)[0][:top_k]
#
#
# # %%
#
# retrieve(query, top_k=10, index=embeddings)