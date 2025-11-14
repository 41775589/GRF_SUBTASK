"""
Chainball Environment Wrapper for EPyMARL
"""
import numpy as np
import chainball.chainball as chainball_env
import numpy as np
from ..multiagentenv import MultiAgentEnv


class ChainballWrapper(MultiAgentEnv):
    """EPyMARL compatible wrapper for Chainball environment"""

    def __init__(self, **kwargs):
        # 从 kwargs 中提取配置
        self.episode_limit = kwargs.get("episode_limit", 100)

        # 创建 Chainball 环境
        env_config = {
            "max_cycles": self.episode_limit,
            "num_agents": kwargs.get("n_agents", 2),
            "num_actions": kwargs.get("n_actions", 4),
            "chain_length": kwargs.get("chain_length", 7),
            "spawn_location": kwargs.get("spawn_location", 4),
            "fixed_time": kwargs.get("fixed_time", False),
            "terminal_states": kwargs.get("terminal_states", []),
            "optimals": kwargs.get("optimals", None),
            "enable_our_goal": kwargs.get("enable_our_goal", True),
            "enable_opp_goal": kwargs.get("enable_opp_goal", True),
            "obs_mode": kwargs.get("obs_mode", "onehot"),
            "normalize_reward": kwargs.get("normalize_reward", False),
            "beta_a": kwargs.get("beta_a", 1),
            "beta_b": kwargs.get("beta_b", 1),
            "optimal_level": kwargs.get("optimal_level", 0.8),
            "noise_level": kwargs.get("noise_level", 0.5),
            "default_back_coef": kwargs.get("default_back_coef", 1.5),
            "env_gen_seed": kwargs.get("env_gen_seed", 1),
        }

        self.env = chainball_env.parallel_env(**env_config)
        self.n_agents = env_config["num_agents"]
        self.n_actions = env_config["num_actions"]

        # 获取观察空间维度
        agent_name = self.env.possible_agents[0]
        obs_space = self.env.observation_space(agent_name)

        if env_config["obs_mode"] == "discrete":
            self.obs_size = 1
        elif env_config["obs_mode"] == "index_box":
            self.obs_size = 1
        elif env_config["obs_mode"] in ["onehot", "onehot_binary"]:
            self.obs_size = obs_space.shape[0]

        # EPyMARL 需要的属性
        self.episode_limit = self.episode_limit
        self._episode_steps = 0
        self._episode_ended = False
        self._total_episodes = 0

    def step(self, actions):
        """
        执行一步
        Args:
            actions: list of int, 每个智能体的动作
        Returns:
            reward: float, 团队奖励
            terminated: bool, 是否终止
            info: dict, 额外信息
        """
        self._episode_steps += 1

        # 将动作列表转换为字典
        action_dict = {
            agent: actions[i]
            for i, agent in enumerate(self.env.agents)
        }

        # 执行环境步骤
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)

        # EPyMARL 使用团队奖励
        reward = sum(rewards.values()) if rewards else 0.0

        # 检查是否结束
        terminated = any(terminations.values()) if terminations else False
        truncated = any(truncations.values()) if truncations else False

        # 检查是否达到时间限制
        if self._episode_steps >= self.episode_limit:
            truncated = True

        self._episode_ended = terminated or truncated

        info = {
            "episode_limit": self._episode_steps >= self.episode_limit,
        }

        return reward, self._episode_ended, info

    def get_obs(self):
        """
        返回所有智能体的观察
        Returns:
            list of observations, shape: [n_agents, obs_size]
        """
        if not self.env.agents:
            # 如果环境已经结束，返回零观察
            return [np.zeros(self.obs_size, dtype=np.float32) for _ in range(self.n_agents)]

        obs_list = []
        for agent in self.env.possible_agents:
            if agent in self.env.agents:
                # 获取当前观察
                obs = self.env.loc_obs(self.env.loc)
            else:
                # 已终止的智能体返回零观察
                obs = np.zeros(self.obs_size, dtype=np.float32)

            # 确保是正确的格式和数据类型
            if isinstance(obs, (int, np.integer)):
                obs = np.array([float(obs)], dtype=np.float32)
            elif isinstance(obs, np.ndarray):
                obs = obs.astype(np.float32)
                # 确保是 1D 数组
                if obs.ndim == 0:
                    obs = np.array([float(obs)], dtype=np.float32)
                elif obs.ndim > 1:
                    obs = obs.flatten()
            else:
                obs = np.array(obs, dtype=np.float32)

            # 检查 NaN 和 Inf
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"警告: 观察包含 NaN 或 Inf，使用零向量替代")
                obs = np.zeros(self.obs_size, dtype=np.float32)

            # 确保维度正确
            if obs.shape[0] != self.obs_size:
                print(f"警告: 观察维度不匹配 {obs.shape[0]} != {self.obs_size}")
                obs = np.zeros(self.obs_size, dtype=np.float32)

            obs_list.append(obs)

        return obs_list

    def get_obs_agent(self, agent_id):
        """
        返回单个智能体的观察
        Args:
            agent_id: int, 智能体ID
        Returns:
            observation array
        """
        obs = self.get_obs()
        return obs[agent_id]

    def get_obs_size(self):
        """返回观察空间的维度"""
        return self.obs_size

    def get_state(self):
        """
        返回全局状态（所有智能体可以看到相同的状态）
        对于 Chainball，全局状态就是球的位置
        Returns:
            state array
        """
        # 使用 one-hot 编码表示位置
        state = np.zeros(self.env.chain_length + 2, dtype=np.float32)
        if hasattr(self.env, 'loc'):
            if 0 <= self.env.loc < len(state):
                state[self.env.loc] = 1.0
        return state

    def get_state_size(self):
        """返回全局状态的维度"""
        return self.env.chain_length + 2

    def get_avail_actions(self):
        """
        返回所有智能体的可用动作
        Returns:
            list of available actions, shape: [n_agents, n_actions]
        """
        # Chainball 中所有动作始终可用
        return [[1] * self.n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """
        返回单个智能体的可用动作
        Args:
            agent_id: int, 智能体ID
        Returns:
            list of available actions
        """
        return [1] * self.n_actions

    def get_total_actions(self):
        """返回动作空间的大小"""
        return self.n_actions

    def reset(self):
        """
        重置环境
        Returns:
            initial observations
        """
        self._episode_steps = 0
        self._episode_ended = False
        self._total_episodes += 1

        # 重置环境
        obs_dict = self.env.reset()

        return self.get_obs()

    def render(self):
        """渲染环境"""
        self.env.render()

    def close(self):
        """关闭环境"""
        self.env.close()

    def seed(self, seed=None):
        """设置随机种子"""
        return self.env.seed(seed)

    def save_replay(self):
        """EPyMARL 可能需要的方法"""
        pass

    def get_env_info(self):
        """
        返回环境信息（EPyMARL 需要）
        Returns:
            dict with environment information
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

    def get_stats(self):
        """
        返回环境统计信息
        Returns:
            dict with statistics
        """
        return {}


#
#
# class ChainballWrapper(MultiAgentEnv):
#     def __init__(self, **kwargs):
#         self.episode_limit = kwargs.get("episode_limit", 100)
#
#         env_config = {
#             "max_cycles": self.episode_limit,
#             "num_agents": kwargs.get("n_agents", 2),
#             "num_actions": kwargs.get("n_actions", 4),
#             "chain_length": kwargs.get("chain_length", 7),
#             "spawn_location": kwargs.get("spawn_location", 4),
#             "fixed_time": kwargs.get("fixed_time", False),
#             "terminal_states": kwargs.get("terminal_states", []),
#             "optimals": kwargs.get("optimals", None),
#             "enable_our_goal": kwargs.get("enable_our_goal", True),
#             "enable_opp_goal": kwargs.get("enable_opp_goal", True),
#             "obs_mode": kwargs.get("obs_mode", "onehot"),
#             "normalize_reward": kwargs.get("normalize_reward", True),
#             "beta_a": kwargs.get("beta_a", 1),
#             "beta_b": kwargs.get("beta_b", 1),
#             "optimal_level": kwargs.get("optimal_level", 0.8),
#             "noise_level": kwargs.get("noise_level", 0.5),
#             "default_back_coef": kwargs.get("default_back_coef", 1.5),
#             "env_gen_seed": kwargs.get("env_gen_seed", 1),
#         }
#
#         # 创建环境
#         self.env = chainball_env.parallel_env(**env_config)
#         self.n_agents = len(self.env.possible_agents)
#         self.n_actions = self.env.num_actions
#         self.obs_mode = self.env.obs_mode
#
#         obs_space = self.env.observation_space(self.env.possible_agents[0])
#         self.obs_shape = int(np.prod(obs_space.shape))
#         self.state_shape = self.obs_shape * self.n_agents
#
#         self.reset()
#
#     def reset(self):
#         obs = self.env.reset()
#         self.steps = 0
#         self._last_individual_rewards = [0.0 for _ in range(self.n_agents)]
#         return self._obs_dict_to_list(obs)
#
#     def _obs_dict_to_list(self, obs_dict):
#         return [obs_dict[a] for a in self.env.possible_agents]
#
#     def step(self, actions):
#         action_dict = {agent: actions[i] for i, agent in enumerate(self.env.possible_agents)}
#         obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
#         self.steps += 1
#
#         terminated = any(terminations.values()) or any(truncations.values())
#         reward_list = [rewards[a] for a in self.env.possible_agents]
#
#         # ====== 安全 reward 归一化 ======
#         if self.env.normalize_reward:
#             min_return = getattr(self.env, "min_return", 0.0)
#             max_return = getattr(self.env, "max_return", 1.0)
#             eps = 1e-8
#             if max_return - min_return < eps:
#                 reward_list = [0.0 for _ in reward_list]
#             else:
#                 reward_list = [(r - min_return) / (max_return - min_return + eps)
#                                for r in reward_list]
#         # ===============================
#
#         reward = np.mean(reward_list)
#         self._last_obs = self._obs_dict_to_list(obs)
#         self._last_individual_rewards = reward_list
#
#         env_info = {"terminated": terminated, "steps": self.steps}
#         return reward, terminated, env_info
#
#     def get_obs(self):
#         return [self.env.loc_obs(self.env.loc) for _ in self.env.possible_agents]
#
#     def get_obs_agent(self, agent_id):
#         return self.env.loc_obs(self.env.loc)
#
#     def get_obs_size(self):
#         return self.obs_shape
#
#     def get_state(self):
#         obs = self.get_obs()
#         return np.concatenate(obs, axis=-1)
#
#     def get_state_size(self):
#         return self.state_shape
#
#     def get_avail_actions(self):
#         return [np.ones(self.n_actions) for _ in range(self.n_agents)]
#
#     def get_avail_agent_actions(self, agent_id):
#         return np.ones(self.n_actions)
#
#     def get_total_actions(self):
#         return self.n_actions
#
#     def get_env_info(self):
#         return {
#             "state_shape": self.get_state_size(),
#             "obs_shape": self.get_obs_size(),
#             "n_actions": self.n_actions,
#             "n_agents": self.n_agents,
#             "episode_limit": self.episode_limit,
#         }
#
#     def render(self):
#         self.env.render()
#
#     def close(self):
#         self.env.close()
#
#     def get_stats(self):
#         return {
#             "steps": getattr(self, "steps", 0),
#             "last_individual_rewards": getattr(self, "_last_individual_rewards", None),
#         }
