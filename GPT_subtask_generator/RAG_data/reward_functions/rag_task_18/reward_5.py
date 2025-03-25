import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward to focus on improving transitions and 
    pace management by central midfield players. It aims to minimize chaotic transitions, 
    encourage proper spacing and timing, and optimize passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to control the pacing and transitions rewards
        self.pace_threshold = 0.1  # Encourages controlled movement speed
        self.spacing_reward = 0.05  # Rewards for optimal spacing in transitions
        self.passing_accuracy_reward = 0.1  # For accurate passes in transition

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        modified_reward = reward.copy()
        components = {"base_score_reward": reward.copy(),
                      "pace_reward": [0.0] * len(reward),
                      "spacing_reward": [0.0] * len(reward),
                      "passing_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Iterate over all agents observations to modify rewards
        for index, obs in enumerate(observation):
            # Reward for speed control to ensure smooth pacing
            if np.linalg.norm(obs['left_team_direction'][obs['active']]) < self.pace_threshold:
                components['pace_reward'][index] = self.pacing_reward
                modified_reward[index] += self.pacing_reward
            
            # Reward for maintaining optimal spacing during transitions
            if 'left_team_roles' in obs and obs['left_team_roles'][obs['active']] == 5:  # Only for central midfield
                for i, teammate_pos in enumerate(obs['left_team']):
                    if i != obs['active'] and 0.1 < np.linalg.norm(teammate_pos - obs['left_team'][obs['active']]) < 0.3:
                        components['spacing_reward'][index] += self.spacing_reward
                
                modified_reward[index] += np.sum(components['spacing_reward'][index])
            
            # Reward for accurate passing in midfield transitions
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                ball_direction = np.linalg.norm(obs['ball_direction'][:2])
                if ball_direction > 0.5: # assuming a pass is made
                    components['passing_accuracy_reward'][index] = self.passing_accuracy_reward
                    modified_reward[index] += self.passing_accuracy_reward
        
        return modified_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
