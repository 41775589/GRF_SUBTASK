import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for offensive strategy tasks, including dribbling, passing, and accurate shooting."""
    
    def __init__(self, env):
        super().__init__(env)
        # Parameters for additional rewards
        self._dribbling_reward = 0.2
        self._passing_reward = 0.3
        self._shooting_reward = 0.5
        self._possession_reward = 0.1

    def reset(self):
        """Resets the environment and any internal state."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Potential state preservation logic goes here."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Potential logic for state restoration goes here."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Defines a new reward function to encourage dribbling, passing, and accurate shooting."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward),
            "possession_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for maintaining possession of the ball
            if o['ball_owned_team'] == 1:
                if o['ball_owned_player'] == o['active']:
                    components["possession_reward"][rew_index] = self._possession_reward
                    reward[rew_index] += components["possession_reward"][rew_index]

            # Rewards for actions related to offensive skills
            if o['sticky_actions'][8] == 1:
                # Dribbling action
                components["dribbling_reward"][rew_index] = self._dribbling_reward
            elif np.any([o['sticky_actions'][0], o['sticky_actions'][1], o['sticky_actions'][2], 
                         o['sticky_actions'][3], o['sticky_actions'][4], o['sticky_actions'][5], o['sticky_actions'][6], o['sticky_actions'][7]]):
                # Passing actions
                components["passing_reward"][rew_index] = self._passing_reward
            
            reward[rew_index] += components["dribbling_reward"][rew_index] + components["passing_reward"][rew_index]

            # Reward for shooting towards the goal
            goal_distance = np.linalg.norm(o['ball'] - np.array([1, 0.044]))  # Assumption: shooting towards right goal
            if goal_distance < 0.1 and o['ball_direction'][0] > 0:
                components["shooting_reward"][rew_index] = self._shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Steps through the environment, modifies the reward, and adds reward components to info."""
        observation, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        
        # Add final reward and components to the info dict
        info['final_reward'] = sum(modified_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, modified_reward, done, info
