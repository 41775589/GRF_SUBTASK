import sys
import os
import numpy as np
import gym
import torch as th
from ..multiagentenv import MultiAgentEnv

# 确保使用本地overcooked
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

#
# class OvercookedWrapper(MultiAgentEnv):
#     def __init__(
#             self,
#             layout_name="cramped_room",
#             horizon=400,
#             seed=0,
#             # dense_reward=True,
#             render=False,
#             num_agents=2,
#             **kwargs
#     ):
#         self.layout_name = layout_name
#         self.episode_limit = horizon
#         self.time_step = 0
#         # self.dense_reward = dense_reward
#         self.render_mode = render
#         self.seed_value = seed
#
#         # 导入overcooked模块
#         try:
#             from overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
#             from overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as BaseOvercookedEnv
#             from overcooked.overcooked_ai_py.mdp.actions import Action
#         except ImportError:
#             try:
#                 from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
#                 from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as BaseOvercookedEnv
#                 from overcooked_ai_py.mdp.actions import Action
#             except ImportError as e:
#                 raise e
#
#
#         # 创建Overcooked环境
#         mdp = OvercookedGridworld.from_layout_name(layout_name)
#         self.env = BaseOvercookedEnv.from_mdp(mdp, horizon=horizon)
#         self.mdp = mdp
#
#         # 设置环境参数
#         self.n_agents = num_agents  # Overcooked always has 2 agents
#
#         # 获取动作数量
#         try:
#             self.n_actions = len(Action.ALL_ACTIONS)  # [STAY, UP, DOWN, LEFT, RIGHT, INTERACT]
#         except:
#             self.n_actions = 6  # Default Overcooked action count
#
#         # 创建动作空间 (仿照GoogleFootballEnv的格式)
#         self.action_space = [gym.spaces.Discrete(self.n_actions) for _ in range(self.n_agents)]
#
#         # 计算观察空间大小
#         obs_size = self._calculate_obs_size()
#         self.observation_space = [
#             gym.spaces.Box(
#                 low=-np.inf,
#                 high=np.inf,
#                 shape=(obs_size,),
#                 dtype=np.float32
#             ) for _ in range(self.n_agents)
#         ]
#
#
#         # 初始化观察
#         self.obs = None
#
#     def step(self, _actions):
#         """Returns reward, terminated, info."""
#         if th.is_tensor(_actions):
#             actions = _actions.cpu().numpy()
#         else:
#             actions = _actions
#
#         self.time_step += 1
#
#         # 转换动作格式
#         joint_action = self._convert_actions(actions)
#
#         try:
#             # 执行动作
#             # print("action:",joint_action)
#             next_state, reward, done, info = self.env.step(joint_action)
#             print(f"Raw reward: {reward}")
#             print(f"Info dict: {info}")
#             # print("next_state:",next_state)
#
#             # 确保info是字典且包含数值
#             if not isinstance(info, dict):
#                 info = {}
#
#             # 确保info字典中的值都是数值类型，避免None值
#             cleaned_info = {}
#             for key, value in info.items():
#                 if value is None:
#                     cleaned_info[key] = 0
#                 elif isinstance(value, (int, float)):
#                     cleaned_info[key] = value
#                 else:
#                     # 尝试转换为数值，失败则设为0
#                     try:
#                         cleaned_info[key] = float(value)
#                     except (TypeError, ValueError):
#                         cleaned_info[key] = 0
#
#             # 更新观察
#             self.obs = self._get_observations()
#
#             # 检查时间限制
#             if self.time_step >= self.episode_limit:
#                 done = True
#
#             # 确保reward是数值类型
#             if reward is None:
#                 reward = 0.0
#             elif not isinstance(reward, (int, float)):
#                 try:
#                     reward = float(reward)
#                 except (TypeError, ValueError):
#                     reward = 0.0
#
#             return reward, done, cleaned_info
#
#         except Exception as e:
#             self.obs = np.zeros((self.n_agents, self.get_obs_size()), dtype=np.float32)
#             return 0.0, True, {}
#
#     def get_obs(self):
#         """Returns all agent observations in a list."""
#         if self.obs is None:
#             self.obs = self._get_observations()
#         return self.obs
#
#     def get_obs_agent(self, agent_id):
#         """Returns observation for agent_id."""
#         obs = self.get_obs()
#         return obs[agent_id]
#
#     def get_obs_size(self):
#         """Returns the size of the observation."""
#         return self._calculate_obs_size()
#
#     def get_global_state(self):
#         """Returns the global state (flattened observations)."""
#         obs = self.get_obs()
#         return obs.flatten()
#
#     def get_state(self):
#         """Returns the global state."""
#         return self.get_global_state()
#
#     def get_state_size(self):
#         """Returns the size of the global state."""
#         return self.get_obs_size() * self.n_agents
#
#     def get_avail_actions(self):
#         """Returns the available actions of all agents in a list."""
#         return [[1 for _ in range(self.n_actions)] for _ in range(self.n_agents)]
#
#     def get_avail_agent_actions(self, agent_id):
#         """Returns the available actions for agent_id."""
#         return self.get_avail_actions()[agent_id]
#
#     def get_total_actions(self):
#         """Returns the total number of actions an agent could ever take."""
#         return self.n_actions
#
#     def reset(self):
#         """Returns initial observations and states."""
#         self.time_step = 0
#
#         try:
#             self.env.reset()
#             self.obs = self._get_observations()
#         except Exception as e:
#             self.obs = np.zeros((self.n_agents, self.get_obs_size()), dtype=np.float32)
#
#         return self.get_obs(), self.get_global_state()
#
#     def render(self):
#         """Render the environment."""
#         if self.render_mode and hasattr(self.env, 'render'):
#             return self.env.render()
#
#     def close(self):
#         """Close the environment."""
#         try:
#             if hasattr(self.env, 'close'):
#                 self.env.close()
#         except Exception as e:
#             pass
#
#     def seed(self, seed=None):
#         """Set random seed."""
#         if seed is not None:
#             self.seed_value = seed
#         np.random.seed(self.seed_value)
#
#     def save_replay(self):
#         """Save a replay."""
#         pass
#
#     def get_stats(self):
#         """Get environment statistics."""
#         return {}
#
#     # ==================== Helper Methods ====================
#
#     def _convert_actions(self, actions):
#         """Convert actions to Overcooked format."""
#         try:
#             from overcooked_ai_py.mdp.actions import Action
#             action_map = Action.ALL_ACTIONS
#         except:
#             # 如果导入失败，使用默认映射
#             action_map = list(range(self.n_actions))
#
#         converted_actions = []
#         for i, action in enumerate(actions[:self.n_agents]):
#             if isinstance(action, (int, np.integer)) and 0 <= action < len(action_map):
#                 converted_actions.append(action_map[action])
#             else:
#                 # 默认动作 (STAY)
#                 converted_actions.append(action_map[0] if action_map else 0)
#
#         return converted_actions
#
#     def _get_observations(self):
#         """Get observations for all agents."""
#         try:
#             obs = np.zeros((self.n_agents, self.get_obs_size()), dtype=np.float32)
#
#             for i in range(self.n_agents):
#                 obs[i] = self._get_agent_observation(i)
#
#             return obs
#         except Exception as e:
#             return np.zeros((self.n_agents, self.get_obs_size()), dtype=np.float32)
#
#     def _get_agent_observation(self, agent_id):
#         """Get observation for a specific agent."""
#         try:
#             state = self.env.state
#             obs = np.zeros(self.get_obs_size(), dtype=np.float32)
#
#             if hasattr(state, 'players') and len(state.players) > agent_id:
#                 player = state.players[agent_id]
#
#                 # Player position (2 features)
#                 if hasattr(player, 'position') and player.position:
#                     # 确保position是可以转换为数字的
#                     pos = player.position
#                     if isinstance(pos, (list, tuple)) and len(pos) >= 2:
#                         obs[0] = float(pos[0])
#                         obs[1] = float(pos[1])
#                     else:
#                         obs[0] = obs[1] = 0.0
#
#                 # Player orientation (1 feature)
#                 if hasattr(player, 'orientation'):
#                     try:
#                         obs[2] = float(player.orientation)
#                     except (TypeError, ValueError):
#                         obs[2] = 0.0
#
#                 # Player held object (1 feature)
#                 if hasattr(player, 'held_object') and player.held_object:
#                     obs[3] = 1.0
#                 else:
#                     obs[3] = 0.0
#
#                 # Other player position (2 features)
#                 other_player_id = 1 - agent_id
#                 if len(state.players) > other_player_id:
#                     other_player = state.players[other_player_id]
#                     if hasattr(other_player, 'position') and other_player.position:
#                         pos = other_player.position
#                         if isinstance(pos, (list, tuple)) and len(pos) >= 2:
#                             obs[4] = float(pos[0])
#                             obs[5] = float(pos[1])
#                         else:
#                             obs[4] = obs[5] = 0.0
#
#                     if hasattr(other_player, 'orientation'):
#                         try:
#                             obs[6] = float(other_player.orientation)
#                         except (TypeError, ValueError):
#                             obs[6] = 0.0
#
#                 # Environment state (remaining features)
#                 # Add more environment-specific features here
#                 # For example: pot states, ingredient locations, etc.
#
#             return obs
#
#         except Exception as e:
#             return np.zeros(self.get_obs_size(), dtype=np.float32)
#
#     def _calculate_obs_size(self):
#         """Calculate the observation size."""
#         # Basic observation features:
#         # - Player position: 2
#         # - Player orientation: 1
#         # - Player held object: 1
#         # - Other player position: 2
#         # - Other player orientation: 1
#         # - Environment features: 43 (expandable)
#         return 50  # Adjust based on your specific needs
#
#     def get_env_info(self):
#         """Get environment info (for EPyMARL compatibility)."""
#         return {
#             "state_shape": self.get_state_size(),
#             "obs_shape": self.get_obs_size(),
#             "n_actions": self.get_total_actions(),
#             "n_agents": self.n_agents,
#             "episode_limit": self.episode_limit
#         }


import random
import os
from typing import Tuple, Any, Dict

import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import TimeLimit as GymTimeLimit
from gymnasium.utils.step_api_compatibility import step_api_compatibility



from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe, pos_distance

# class CustomMDP(OvercookedGridworld):
#     def resolve_interacts(self, new_state, joint_action, events_infos):
#         # 保留 base 行为（会返回 per-agent 的 sparse 和 shaped 列表）
#         sparse, shaped = super().resolve_interacts(new_state, joint_action, events_infos)
#
#         # 举例：给靠近锅的玩家一点额外 shaped 奖励（越近奖励越高）
#         pot_positions = [p for p in self.get_pot_positions()]  # helper 在不同版本名可能不同
#         for i, player in enumerate(new_state.players):
#             dmin = min(pos_distance(player.position, pot) for pot in pot_positions)
#             shaped[i] += max(0.0, 1.0 - 0.2 * dmin)  # 自定义系数
#
#         return sparse, shaped
#
#     def deliver_soup(self, state, player, soup):
#         # 可以先调用父类得到基础上菜奖励，然后再加自定义 bonus
#         base_reward = super().deliver_soup(state, player, soup)
#         if some_custom_condition(soup, state):
#             base_reward += 10
#         return base_reward

def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# class IngredientSpecialistMDP(OvercookedGridworld):
#     """食材准备专家子任务MDP - 专注于Agent 0的食材处理任务"""
#
#     def __init__(self, base_layout_name="cramped_room", **kwargs):
#         super().__init__(base_layout_name, **kwargs)
#         self.specialist_agent_id = 0  # Agent 0是食材专家
#         self.previous_onions_in_pot = 0
#         self.previous_agent_had_onion = False
#
#
#     def resolve_interacts(self, new_state, joint_action, events_infos):
#         """为食材专家提供详细的奖励塑形"""
#         # 获取基础奖励
#         sparse, shaped = super().resolve_interacts(new_state, joint_action, events_infos)
#
#         specialist = new_state.players[self.specialist_agent_id]
#         assistant = new_state.players[1 - self.specialist_agent_id]
#
#         # =============食材专家奖励塑形=============
#
#         # 1. 洋葱相关奖励
#         if specialist.held_object and hasattr(specialist.held_object, 'name'):
#             obj_name = str(specialist.held_object.name).lower()
#
#             # 持有洋葱的基础奖励
#             if 'onion' in obj_name:
#                 shaped[self.specialist_agent_id] += 0.1
#
#                 # 持有洋葱时靠近锅的额外奖励
#                 pot_positions = self.get_pot_locations()
#                 if pot_positions:
#                     min_pot_dist = min(manhattan_distance(specialist.position, pot)
#                                        for pot in pot_positions)
#                     if min_pot_dist == 1:  # 相邻锅
#                         shaped[self.specialist_agent_id] += 0.3
#                     elif min_pot_dist == 2:  # 较近
#                         shaped[self.specialist_agent_id] += 0.1
#
#         # 2. 靠近洋葱储存区的奖励（当没有持有物品时）
#         if not specialist.held_object:
#             onion_positions = self.get_onion_dispenser_locations()
#             if onion_positions:
#                 min_onion_dist = min(manhattan_distance(specialist.position, onion)
#                                      for onion in onion_positions)
#                 if min_onion_dist == 1:  # 相邻洋葱储存区
#                     shaped[self.specialist_agent_id] += 0.05
#
#         # 3. 投料奖励 - 检测锅中洋葱数量变化
#         current_onions_in_pot = sum(len(pot.ingredients) for pot in new_state.objects.values()
#                                     if hasattr(pot, 'ingredients'))
#         if current_onions_in_pot > self.previous_onions_in_pot:
#             shaped[self.specialist_agent_id] += 0.5  # 成功投料
#         self.previous_onions_in_pot = current_onions_in_pot
#
#         # 4. 协作奖励 - 避免阻挡助手
#         if manhattan_distance(specialist.position, assistant.position) > 1:
#             shaped[self.specialist_agent_id] += 0.02  # 保持距离
#
#         # 5. 效率惩罚 - 鼓励快速行动
#         shaped[self.specialist_agent_id] -= 0.01
#
#         # =============助手的消极奖励（鼓励让路）=============
#         assistant_id = 1 - self.specialist_agent_id
#
#         # 助手持有餐具时的奖励（准备装盘）
#         if assistant.held_object and hasattr(assistant.held_object, 'name'):
#             if 'dish' in str(assistant.held_object.name).lower():
#                 shaped[assistant_id] += 0.1
#
#         # 助手靠近交付台的奖励
#         serving_positions = self.get_serving_locations()
#         if serving_positions and assistant.held_object:
#             min_serving_dist = min(manhattan_distance(assistant.position, serving)
#                                    for serving in serving_positions)
#             if min_serving_dist == 1:
#                 shaped[assistant_id] += 0.1
#
#         return sparse, shaped
#
#     def deliver_soup(self, state, player, soup):
#         """交付汤时的奖励分配"""
#         base_reward = super().deliver_soup(state, player, soup)
#
#         # 如果是食材专家交付（不太理想），给较少奖励
#         if state.players.index(player) == self.specialist_agent_id:
#             return base_reward * 0.5  # 食材专家不应该负责交付
#         else:
#             # 助手交付，给额外奖励
#             return base_reward * 1.2

class IngredientSpecialistMDP(OvercookedGridworld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specialist_agent_id = 0
        self.previous_onions_in_pot = 0
        self.previous_agent_had_onion = False

    @classmethod
    def from_layout_name(cls, layout_name="cramped_room", **kwargs):
        # 先用父类创建一个 OvercookedGridworld 对象
        base_mdp = OvercookedGridworld.from_layout_name(layout_name, **kwargs)

        # 新建子类实例（传入必要的参数）
        self = cls.__new__(cls)
        self.__dict__.update(base_mdp.__dict__)  # 复制父类实例的所有属性

        # 再加上子类特有的属性
        self.specialist_agent_id = 0
        self.previous_onions_in_pot = 0
        self.previous_agent_had_onion = False
        return self

    def resolve_interacts(self, new_state, joint_action, events_infos):
        """为食材专家提供详细的奖励塑形"""
        # 获取基础奖励
        sparse, shaped = super().resolve_interacts(new_state, joint_action, events_infos)

        specialist = new_state.players[self.specialist_agent_id]
        assistant = new_state.players[1 - self.specialist_agent_id]

        # =============食材专家奖励塑形=============

        # 1. 洋葱相关奖励
        if specialist.held_object and hasattr(specialist.held_object, 'name'):
            obj_name = str(specialist.held_object.name).lower()

            # 持有洋葱的基础奖励
            if 'onion' in obj_name:
                shaped[self.specialist_agent_id] += 0.1

                # 持有洋葱时靠近锅的额外奖励
                pot_positions = self.get_pot_locations()
                if pot_positions:
                    min_pot_dist = min(manhattan_distance(specialist.position, pot)
                                       for pot in pot_positions)
                    if min_pot_dist == 1:  # 相邻锅
                        shaped[self.specialist_agent_id] += 0.3
                    elif min_pot_dist == 2:  # 较近
                        shaped[self.specialist_agent_id] += 0.1

        # 2. 靠近洋葱储存区的奖励（当没有持有物品时）
        if not specialist.held_object:
            onion_positions = self.get_onion_dispenser_locations()
            if onion_positions:
                min_onion_dist = min(manhattan_distance(specialist.position, onion)
                                     for onion in onion_positions)
                if min_onion_dist == 1:  # 相邻洋葱储存区
                    shaped[self.specialist_agent_id] += 0.05

        # 3. 投料奖励 - 检测锅中洋葱数量变化
        current_onions_in_pot = sum(len(pot.ingredients) for pot in new_state.objects.values()
                                    if hasattr(pot, 'ingredients'))
        if current_onions_in_pot > self.previous_onions_in_pot:
            shaped[self.specialist_agent_id] += 0.5  # 成功投料
        self.previous_onions_in_pot = current_onions_in_pot

        # 4. 协作奖励 - 避免阻挡助手
        if manhattan_distance(specialist.position, assistant.position) > 1:
            shaped[self.specialist_agent_id] += 0.02  # 保持距离

        # 5. 效率惩罚 - 鼓励快速行动
        shaped[self.specialist_agent_id] -= 0.01

        # =============助手的消极奖励（鼓励让路）=============
        assistant_id = 1 - self.specialist_agent_id

        # 助手持有餐具时的奖励（准备装盘）
        if assistant.held_object and hasattr(assistant.held_object, 'name'):
            if 'dish' in str(assistant.held_object.name).lower():
                shaped[assistant_id] += 0.1

        # 助手靠近交付台的奖励
        serving_positions = self.get_serving_locations()
        if serving_positions and assistant.held_object:
            min_serving_dist = min(manhattan_distance(assistant.position, serving)
                                   for serving in serving_positions)
            if min_serving_dist == 1:
                shaped[assistant_id] += 0.1

        return sparse, shaped

    def deliver_soup(self, state, player, soup):
        """交付汤时的奖励分配"""
        base_reward = super().deliver_soup(state, player, soup)

        # 如果是食材专家交付（不太理想），给较少奖励
        if state.players.index(player) == self.specialist_agent_id:
            return base_reward * 0.5  # 食材专家不应该负责交付
        else:
            # 助手交付，给额外奖励
            return base_reward * 1.2


class ServiceSpecialistMDP(OvercookedGridworld):
    """服务配送专家子任务MDP - 专注于Agent 1的服务任务"""

    def __init__(self, base_layout_name="cramped_room", **kwargs):
        super().__init__(base_layout_name, **kwargs)
        self.specialist_agent_id = 1  # Agent 1是服务专家
        self.previous_dishes_picked = 0
        self.soup_ready_steps = 0

    def resolve_interacts(self, new_state, joint_action, events_infos):
        """为服务专家提供详细的奖励塑形"""
        # 获取基础奖励
        sparse, shaped = super().resolve_interacts(new_state, joint_action, events_infos)

        specialist = new_state.players[self.specialist_agent_id]
        assistant = new_state.players[1 - self.specialist_agent_id]

        # =============服务专家奖励塑形=============

        # 1. 餐具管理奖励
        if specialist.held_object and hasattr(specialist.held_object, 'name'):
            obj_name = str(specialist.held_object.name).lower()

            # 持有盘子的基础奖励
            if 'dish' in obj_name:
                shaped[self.specialist_agent_id] += 0.1

                # 持有盘子时靠近已完成汤的奖励
                pot_positions = self.get_pot_locations()
                for pot_pos in pot_positions:
                    # 检查这个位置是否有完成的汤
                    pot_obj = new_state.get_object(pot_pos)
                    if pot_obj and hasattr(pot_obj, 'is_ready') and pot_obj.is_ready:
                        dist = manhattan_distance(specialist.position, pot_pos)
                        if dist == 1:  # 相邻已完成的汤
                            shaped[self.specialist_agent_id] += 0.4
                        elif dist == 2:  # 较近
                            shaped[self.specialist_agent_id] += 0.2

            # 持有装好汤的盘子时靠近交付台的奖励
            elif 'soup' in obj_name:
                serving_positions = self.get_serving_locations()
                if serving_positions:
                    min_serving_dist = min(manhattan_distance(specialist.position, serving)
                                           for serving in serving_positions)
                    if min_serving_dist == 1:  # 相邻交付台
                        shaped[self.specialist_agent_id] += 0.6
                    elif min_serving_dist == 2:  # 较近
                        shaped[self.specialist_agent_id] += 0.3

        # 2. 靠近盘子储存区的奖励（当没有持有物品时）
        if not specialist.held_object:
            dish_positions = self.get_dish_dispenser_locations()
            if dish_positions:
                min_dish_dist = min(manhattan_distance(specialist.position, dish)
                                    for dish in dish_positions)
                if min_dish_dist == 1:  # 相邻盘子储存区
                    shaped[self.specialist_agent_id] += 0.08

        # 3. 等待汤完成的耐心奖励
        ready_soups = sum(1 for pot in new_state.objects.values()
                          if hasattr(pot, 'is_ready') and pot.is_ready)
        if ready_soups > 0 and specialist.held_object:
            if 'dish' in str(specialist.held_object.name).lower():
                shaped[self.specialist_agent_id] += 0.15  # 有汤可装且持有盘子

        # 4. 协作奖励 - 给助手让路
        if assistant.held_object and not specialist.held_object:
            # 助手有物品时，服务专家应该避开
            if manhattan_distance(specialist.position, assistant.position) > 1:
                shaped[self.specialist_agent_id] += 0.03

        # 5. 效率惩罚
        shaped[self.specialist_agent_id] -= 0.01

        # =============助手的支持奖励=============
        assistant_id = 1 - self.specialist_agent_id

        # 助手成功投料时的支持奖励
        if assistant.held_object and hasattr(assistant.held_object, 'name'):
            if 'onion' in str(assistant.held_object.name).lower():
                shaped[assistant_id] += 0.05  # 食材助手的基础奖励

        # 助手不阻挡服务专家的奖励
        if specialist.held_object and 'soup' in str(specialist.held_object.name).lower():
            # 服务专家持有汤时，助手应该让路
            if manhattan_distance(specialist.position, assistant.position) > 1:
                shaped[assistant_id] += 0.05

        return sparse, shaped

    def deliver_soup(self, state, player, soup):
        """交付汤时的奖励分配"""
        base_reward = super().deliver_soup(state, player, soup)

        # 如果是服务专家交付（理想情况），给额外奖励
        if state.players.index(player) == self.specialist_agent_id:
            # 检查交付速度奖励
            speed_bonus = 0
            if hasattr(soup, 'cook_time') and soup.cook_time is not None:
                # 根据烹饪完成后的等待时间给速度奖励
                if soup.cook_time < 10:  # 快速交付
                    speed_bonus = 5
                elif soup.cook_time < 20:  # 中等速度
                    speed_bonus = 2

            return base_reward * 1.5 + speed_bonus  # 服务专家应该负责交付
        else:
            # 食材专家交付，给基础奖励
            return base_reward * 0.8
#######################################################################################################################

class TimeLimitOvercooked(GymTimeLimit):

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env, max_episode_steps=max_episode_steps)

        assert max_episode_steps is not None, "'max_episode_steps' is None!"
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def timelimit_step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

        observation, reward, done, info = step_api_compatibility(self.env.step(action), output_truncation_bool=False)

        self._elapsed_steps += 1
        info["TimeLimit.truncated"] = False  # There is no truncation in Overcooked
        if self._elapsed_steps >= self._max_episode_steps:
            done = True

        return observation, reward, done, info

    def get_elapsed_steps(self):
        return self._elapsed_steps


class ObservationOvercooked(ObservationWrapper):
    """
    Observation wrapper that fixes the order of agents' observations.
    """

    def __init__(self, env):
        super(ObservationOvercooked, self).__init__(env)

        self.observation_space: tuple = env.observation_space.shape
        self.timelimit_env = env
        self.other_agent_idx = None
        self.agent_policy_idx = None

    def observation(self, observation):

        if hasattr(self.timelimit_env, 'get_elapsed_steps'):
            if self.timelimit_env.get_elapsed_steps() == 0:  # Called from reset()
                # Get agents' ids to fix their observations and actions' order
                self.other_agent_idx = observation['other_agent_env_idx']
                self.agent_policy_idx = 1 - self.other_agent_idx
        else:
            raise AttributeError("The 'get_elapsed_steps' method is not implemented in the '_OvercookedWrapper'")

        # Fix the order of observations, 'policy_agent_idx' always corresponds to agent 0
        assert self.agent_policy_idx == 1 - self.other_agent_idx
        assert self.other_agent_idx == observation['other_agent_env_idx']

        ####################################################MODIFY########################################################################
        observation = (
            observation['both_agent_obs'][self.agent_policy_idx],
            observation['both_agent_obs'][self.other_agent_idx]
        )
        # observation = (
        #     observation['both_agent_obs'][self.agent_policy_idx],
        # )
        ####################################################MODIFY########################################################################

        return observation

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if hasattr(self.timelimit_env, 'timelimit_step'):
            observation, reward, done, info = self.timelimit_env.timelimit_step(action)
        else:
            raise AttributeError("The 'timelimit_step' method is not implemented in the '_OvercookedWrapper'")

        return self.observation(observation), reward, done, info


OVERCOOKED_KEY_CHOICES = [
    "random3",
    "random0",
    "unident",
    "soup_coordination",
    "small_corridor",
    "simple_tomato",
    "simple_o_t",
    "simple_o",
    "schelling_s",
    "schelling",
    "m_shaped_s",
    "long_cook_time",
    "large_room",
    "forced_coordination_tomato",
    "forced_coordination",
    "cramped_room_tomato",
    "cramped_room_o_3orders",
    "cramped_room",
    "cramped_corridor",
    "counter_circuit_o_1order",
    "counter_circuit",
    "corridor",
    "coordination_ring",
    "centre_objects",
    "centre_pots",
    "asymmetric_advantages",
    "asymmetric_advantages_tomato",
    "bottleneck"
]
OVERCOOKED_REWARD_TYPE_CHOICES = ["shaped", "sparse"]


class OvercookedWrapper(MultiAgentEnv):

    def __init__(self, layout_name="cramped_room" , time_limit=500, seed=1, reward_type="shaped", num_agents=2, render=False,**kwargs):

        super().__init__()

        # # Check key validity
        # assert layout_name in OVERCOOKED_KEY_CHOICES, \
        #     f"Invalid 'key': {layout_name}! \nChoose one of the following: \n{OVERCOOKED_KEY_CHOICES}"
        # Check time_limit validity
        assert isinstance(time_limit, int), \
            f"Invalid time_limit type: {type(time_limit)}, 'time_limit': {time_limit}, is not 'int'!"
        # Check reward_type validity
        assert reward_type in OVERCOOKED_REWARD_TYPE_CHOICES, \
            f"Invalid 'reward_type': {reward_type}! \nChoose one of the following: \n{OVERCOOKED_REWARD_TYPE_CHOICES}"

        self.layout_name = layout_name
        self._seed = seed  # Just for compatibility since the agents start always from the same position
        self.reward_type = reward_type
        self.render_bool = render

        # Placeholders
        self._obs = None
        self._info = None
        self.internal_print_info = None

        # Check the consistency between the 'render_bool' and the display capabilities of the machine
        self.render_capable = True
        if self.render_bool is True and 'DISPLAY' not in os.environ:
            self.render_bool = False
            self.internal_print_info = (
                "\n\n###########################################################"
                "\nThe 'render' is set to 'False' due to the lack of display capabilities!"
                "\n###########################################################\n"
            )
            self.render_capable = False

        # 导入overcooked模块
        try:
            from overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
            from overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as BaseOvercookedEnv
            from overcooked.overcooked_ai_py.mdp.actions import Action
        except ImportError:
            try:
                from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
                from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as BaseOvercookedEnv
                from overcooked_ai_py.mdp.actions import Action
            except ImportError as e:
                raise e

        # Gymnasium make
        # mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        # mdp = ServiceSpecialistMDP.from_layout_name(self.layout_name)
        mdp = IngredientSpecialistMDP.from_layout_name(self.layout_name)


        print(f"✅ 创建的MDP类型: {type(mdp).__name__}")

        base_env = BaseOvercookedEnv.from_mdp(mdp, horizon=time_limit)
        self.original_env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

        # Use the wrappers for handling the time limit and the environment observations properly.
        self.episode_limit = time_limit
        self.n_agents = 2 # Always 2 agents?
        self.timelimit_env = TimeLimitOvercooked(self.original_env, max_episode_steps=self.episode_limit)
        self._env = ObservationOvercooked(self.timelimit_env)

        # Define the observation space
        self.observation_space: tuple = self._env.observation_space  # type: ignore[override]

        # Define the action space
        if hasattr(self._env.action_space, 'n'):
            self.action_space = self._env.action_space.n
        else:
            raise AttributeError(f"'n' attribute not found in action space in overcooked environment with layout: {layout_name}")

        # Needed for rendering
        import cv2
        self.cv2 = cv2

    def get_print_info(self):
        print_info = self.internal_print_info

        # Clear the internal print info
        self.internal_print_info = None

        return print_info

    def step(self, actions):
        """ Returns reward, terminated, info """

        if self.render_bool is True:
            self.render()

        # Fix the order of actions, 'policy_agent_idx' always corresponds to agent 0
        actions = [int(a) for a in actions]
        if self._env.agent_policy_idx == 1:
            actions = actions[::-1]  # reverse the order

        # Make the environment step
        self._obs, reward, done, self._info = self._env.step(actions)

        if self.reward_type == "shaped":
            assert type(self._info['shaped_r_by_agent']) is list, \
                "'self._info['shaped_r_by_agent']' is not a list! " + \
                f"'self._info['shaped_r_by_agent']': {self._info['shaped_r_by_agent']}"
            reward = sum(self._info['shaped_r_by_agent'])
        # else: the other option is the sum of sparse rewards which is the default 'reward'

        # Keep only 'TimeLimit.truncated' in 'self._info'
        self._info = {"TimeLimit.truncated": self._info["TimeLimit.truncated"]}

        # Handle different cases of 'done'
        if isinstance(done, (list, tuple)):
            done = all(done)
        else:
            assert isinstance(done, bool) and done is True

        return float(reward), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.observation_space[0]

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state """

        assert len(self.observation_space) == 1, \
            f"'self.observation_space' has not only one dimension! \n'self.observation_space': {self.observation_space}"

        return self.n_agents * self.observation_space[0]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id (both agents have the same action space) """
        return self.action_space * [1]  # 1 indicates availability of action

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return int(self.action_space)

    def sample_actions(self):
        return random.choices(range(0, self.get_total_actions()), k=self.n_agents)

    def reset(self, seed=None):
        """ Returns initial observations and states """

        # Randomness does not affect Overcooked, so we don't pass it to the environment
        if seed is not None:
            self._seed = seed

        self._obs, _ = self._env.reset()

        return self.get_obs(), self.get_state()

    def get_info(self):
        return self._info

    def get_n_agents(self):
        return self.n_agents

    def render(self):
        if self.render_capable is True:
            try:
                image = self._env.render()
                image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
                self.cv2.imshow("Overcooked", image)
                self.cv2.waitKey(1)
            except (Exception, SystemExit) as e:
                self.internal_print_info = (
                    "\n\n###########################################################"
                    f"\nError during rendering: \n\n{e}"
                    f"\n\nRendering will be disabled to continue the training."
                    "\n###########################################################\n"
                )
                self.render_capable = False

    def close(self):
        self._env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    @staticmethod
    def get_stats():
        return {}