import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focusing on offensive strategies like accurate shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 0.3
        self.dribbling_reward = 0.2
        self.passing_reward = 0.1
        self.goal_box_x_threshold = 0.7  # when x is greater than 0.7, it is near the opponent's goal box
        self.pass_types = {8: 'high_pass', 9: 'long_pass'}

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
        components = {"base_score_reward": reward.copy()}

        # Add extra rewards
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            # Reward for being close to the opponent's goal with the ball
            if o['ball_owned_team'] == 0 and o['ball'][0] > self.goal_box_x_threshold:
                reward[i] += self.shooting_reward

            # Reward for dribbling (having the ball and the sticky_actions indicating dribbling)
            if o['ball_owned_team'] == 0 and 9 in o['sticky_actions']:  # index 9 is dribbling action
                reward[i] += self.dribbling_reward

            # Reward for performing high or long passes (triggering action 8 or 9 corresponding to pass types)
            for action_type in self.pass_types:
                if action_type in o['sticky_actions']:
                    reward[i] += self.passing_reward

        return reward, {"base_score_reward": components["base_score_reward"]}

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method to implement the scoring logic for offensive strategies
        reward, components = self.reward(reward)
        
        # Add final reward to the info
        info["final_reward"] = sum(reward)
        
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        return observation, reward, done, info
