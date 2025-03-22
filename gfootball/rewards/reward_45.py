import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on offensive gameplay enhancements including shooting accuracy, 
       dribbling to evade opponents, and making effective passes."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_accuracy_bonus = 0.1
        self.dribbling_bonus = 0.05
        self.passing_bonus = 0.08

    def reset(self):
        self.pass_history = [False] * 5  # Arbitrary number for demonstration; adjust based on number of players controlled.
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy": [0.0] * len(reward),
                      "dribbling_bonus": [0.0] * len(reward),
                      "passing_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'sticky_actions' in o:
                if o['sticky_actions'][9] == 1:  # Dribble action is being used.
                    components["dribbling_bonus"][rew_index] = self.dribbling_bonus
                if o['sticky_actions'][8] == 1:  # Sprint action is being used, typically involves aggressive forward movement.
                    components["dribbling_bonus"][rew_index] += self.dribbling_bonus

            if 'ball_owned_player' in o and o['ball_owned_team'] == 0:  # 0 indicates left team; adjust appropriately.
                player_id = o['ball_owned_player']
                
                # Assuming the shooting mechanism, here based on proximity to goal and ball possession.
                if o['ball'][0] > 0.8:  # This parameter might need adjustments based on the actual game coordinates.
                    components["shooting_accuracy"][rew_index] += self.shooting_accuracy_bonus

                # Check for effective passing, simple assumption: Ball changes ownership from one player to another.
                if self.pass_history[rew_index]:
                    components["passing_bonus"][rew_index] += self.passing_bonus
                self.pass_history[rew_index] = player_id  # Update pass history to the last player holding the ball.

            # Sum total rewards for each component added for this step.
            reward[rew_index] += sum(components[k][rew_index] for k in components)
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
