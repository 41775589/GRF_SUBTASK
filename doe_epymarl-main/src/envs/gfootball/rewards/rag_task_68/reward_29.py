import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on offensive strategies in football:
    - Encourages mastering accurate shooting.
    - Rewards effective dribbling to evade opponents.
    - Incentivizes practicing different pass types to break defensive lines.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.5
        self.long_pass_reward = 0.3
        self.high_pass_reward = 0.4
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        """
        Resets the environment and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Returns the state for serialization.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Sets the deserialized state.
        """
        state_info = self.env.set_state(state)
        self.sticky_actions_counter = state_info['sticky_actions_counter']
        return state_info

    def reward(self, reward):
        """
        Modifies the reward function based on possession, dribbling, and successful long/high passes.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, obs in enumerate(observation):
            # Shooting - Looks for active control and goal attempts.
            if obs['game_mode'] == 6:  # Penalty as an approximation for shooting efforts
                components['shooting_reward'][i] = self.shooting_reward
                reward[i] += components['shooting_reward'][i]

            # Effective dribbling - reward for dribbling while avoiding loss of possession.
            if obs['sticky_actions'][9] == 1:  # Checking if dribbling is active
                components['dribbling_reward'][i] = self.dribbling_reward
                reward[i] += components['dribbling_reward'][i]

            # Long passes - promoting usage of long passes to break lines.
            if obs['ball_owned_team'] == 1 and np.linalg.norm(obs['ball_direction'][:2]) > 0.5:
                # Assuming ball_direction magnitude large represents a long/thrust pass
                components['long_pass_reward'][i] = self.long_pass_reward
                reward[i] += components['long_pass_reward'][i]

            # High passes - promoting usage of high passes.
            if obs['ball_owned_team'] == 1 and abs(obs['ball_direction'][2]) > 0.1:
                # Assuming vertical component of the ball direction represents a high pass
                components['high_pass_reward'][i] = self.high_pass_reward
                reward[i] += components['high_pass_reward'][i]

        return reward, components

    def step(self, action):
        """
        Steps through the environment with the provided action, and augments the reward response.
        """
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
