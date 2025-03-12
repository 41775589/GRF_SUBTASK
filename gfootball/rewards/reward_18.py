import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on offensive strategies:
       accurate shooting, effective dribbling, and strategic passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self._shooting_reward = 0.2
        self._dribbling_reward = 0.1
        self._passing_reward = 0.15
        self._goal_scored_reward = 1.0
        
    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        return from_picle
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = { "base_score_reward": reward.copy(),
                       "shooting_reward": [0.0] * len(reward),
                       "dribbling_reward": [0.0] * len(reward),
                       "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            if 'game_mode' in o and o['game_mode'] == 6 and o['ball_owned_team'] == o['active']:
                # Reward for shooting accuracy if it's in a penalty game mode and the ball is owned by player
                components["shooting_reward"][rew_index] = self._shooting_reward
            
            if o['sticky_actions'][9] == 1:  # dribble action is active
                # Reward for effective dribbling
                components["dribbling_reward"][rew_index] = self._dribbling_reward

            if o['game_mode'] in [2, 4, 5] and o['ball_owned_team'] == o['active']:
                # Reward strategic passes (from free-kicks, corners, or throw-ins)
                components["passing_reward"][rew_index] = self._passing_reward

            # Accumulating all rewards
            total_additional_reward = (components["shooting_reward"][rew_index] +
                                       components["dribbling_reward"][rew_index] +
                                       components["passing_reward"][rew_index])
            reward[rew_index] = base_reward + total_additional_reward

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward method
        new_reward, components = self.reward(reward)
        
        # Add final reward to the info
        info["final_reward"] = sum(new_reward)

        # Add component values to info for detailed insights
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, new_reward, done, info
