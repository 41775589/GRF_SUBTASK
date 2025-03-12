import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds additional rewards for offensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 0.2  # Reward for shooting towards the goal
        self.dribbling_reward = 0.1  # Reward for dribbling near opponents
        self.passing_reward = 0.15   # Reward for effective passing

    def reset(self):
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
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            
            # Check for shots towards goal.
            if 'ball_direction' in o:
                ball_speed = np.linalg.norm(o['ball_direction'][:2])
                goal_direction = np.sign(o['ball'][0])  # Assuming goal is at x=±1
                if o['ball_owned_team'] == 0 and np.sign(o['ball'][0]) == goal_direction:
                    if ball_speed > 0.1:  # assuming some threshold for a shot attempt
                        reward[i] += self.shooting_reward
                        components["shooting_reward"][i] = self.shooting_reward

            # Reward dribbling within close proximity to opponents
            player_pos = o['left_team'][0] if o['ball_owned_team'] == 0 else o['right_team'][0]
            opponents = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
            distances = np.linalg.norm(opponents - player_pos, axis=1)
            close_opponents = distances < 0.1  # arbitrary distance threshold
            
            if any(close_opponents) and o['sticky_actions'][9] == 1:  # action 9 is dribbling
                reward[i] += self.dribbling_reward
                components["dribbling_reward"][i] = self.dribbling_reward

            # Reward successful long or high passes
            if 'ball_direction' in o and o['ball_owned_team'] == 0:
                ball_distance = np.linalg.norm(o['ball_direction'][:2])
                ball_height = o['ball'][2]
                
                if ball_distance > 0.5 and ball_height > 0.1:  # arbitrary thresholds for a "long high pass"
                    reward[i] += self.passing_reward
                    components["passing_reward"][i] = self.passing_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Adding components and final reward to info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
