import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on offensive strategies: accurate shooting,
    effective dribbling, and utilizing long/high passes.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.5
        self.passing_reward = 0.3
        self.shot_threshold = 0.1 # Threshold distance to goal for considering a 'shot'
        self.pass_types_rewards = {
            'long': self.passing_reward,
            'high': self.passing_reward
        }
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Criteria for shooting reward: close to goal and shoot action triggered
            if 'ball' in o:
                distance_to_goal = abs(1 - o['ball'][0])  # 1 is the x-coordinate for opponent's goal
                if distance_to_goal < self.shot_threshold and self.sticky_actions_counter[4] == 1:  # Action 4 corresponds to 'shoot'
                    components["shooting_reward"][rew_index] += self.shooting_reward
                    reward[rew_index] += components["shooting_reward"][rew_index]

            # Criteria for dribbling reward: dribble action is ongoing
            if self.sticky_actions_counter[9] == 1:  # Action 9 corresponds to 'dribble'
                components["dribbling_reward"][rew_index] += self.dribbling_reward
                reward[rew_index] += components["dribbling_reward"][rew_index]

            # Example for passing reward: specific pass types could be detected by some heuristic or predefined conditions
            if 'game_mode' in o:
                if o['game_mode'] in [3, 4, 5]:  # Simplified example: FreeKick, Corner or ThrowIn can be assumed to involve a strategic pass
                    pass_type = 'long' if o['game_mode'] == 5 else 'high'
                    pass_reward = self.pass_types_rewards.get(pass_type, 0)
                    components["passing_reward"][rew_index] += pass_reward
                    reward[rew_index] += components["passing_reward"][rew_index]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
