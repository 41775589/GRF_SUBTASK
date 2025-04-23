import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward for mastering defensive tactics with a focus
    on executing both sliding and standing tackles without fouling.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def reward(self, reward):
        # Access the observation from the environment
        observation = self.env.unwrapped.observation()
        
        # Initialize reward components
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "foul_penalty": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] == 6:  # Operating in Penalty (fouls typically occur here)
                components['foul_penalty'][rew_index] -= 1.0
            if o['sticky_actions'][6] or o['sticky_actions'][7]:  # Check if sliding or standing tackle is used
                if o['ball_owned_team'] == 1:  # Ensuring the ball is owned by opponents
                    components['tackle_reward'][rew_index] = 0.1
                else:
                    components['foul_penalty'][rew_index] -= 0.5  # Penalize potentially wrong tackle
        
        # Compute the overall reward by summing up the components
        for rew_index in range(len(reward)):
            reward[rew_index] += components['tackle_reward'][rew_index]
            reward[rew_index] += components['foul_penalty'][rew_index]
        
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle
