import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for midfield dynamics mastery and transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Parameters for midfield dynamics
        self.midfield_reward = 0.1
        self.transition_bonus = 0.2
        
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
                      "midfield_reward": [0.0] * len(reward), 
                      "transition_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            ball_position_x = o['ball'][0]

            # Reward for maintaining position in the midfield
            if -0.3 <= ball_position_x <= 0.3:
                components["midfield_reward"][i] = self.midfield_reward
                reward[i] += components["midfield_reward"][i]
            
            # Bonus for transitioning from defense to offense or vice versa effectively
            if o['game_mode'] in {1, 3, 4, 5, 6}:  # transitions game modes
                components["transition_bonus"][i] = self.transition_bonus
                reward[i] += components["transition_bonus"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        return observation, reward, done, info
