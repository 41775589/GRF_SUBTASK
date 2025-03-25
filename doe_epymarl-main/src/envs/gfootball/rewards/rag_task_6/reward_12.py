import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes energy conservation through strategic use of 
    Stop-Sprint and Stop-Moving actions, crucial for maintaining stamina 
    and strategic positional play over the duration of a match.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.stamina_threshold = 0.2  # Threshold to encourage less sprint usage when low
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "stamina_management_reward": [0.0] * len(reward)}
        
        # Adjust reward based on stamina management
        for i, obs in enumerate(observation):
            if obs['sticky_actions'][8] == 0:  # Not sprinting
                # Provide a reward for not sprinting when stamina is low
                if obs['left_team_tired_factor'][obs['active']] > self.stamina_threshold or \
                   obs['right_team_tired_factor'][obs['active']] > self.stamina_threshold:
                    components["stamina_management_reward"][i] = 0.1
                    reward[i] += components["stamina_management_reward"][i]

            # Provide reward for stopping (relevant action modes not used)
            if all(action == 0 for action in obs['sticky_actions'][:6]):  # No directional movement
                components["stamina_management_reward"][i] += 0.05
                reward[i] += components["stamina_management_reward"][i]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
