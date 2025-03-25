import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides enhanced rewards for executing high passes with precision."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions_counter', 
                                                      np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_precision": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]

            if o is None or 'ball_owned_team' not in o:
                continue
            
            if o['ball_owned_team'] == 0 and 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:  # Check if controlled player owns the ball
                # Check for high trajectory (z > 0.15 can be considered as attempting a high pass)
                if o['ball'][2] > 0.15:    
                    # Calculate the power and efficiency of the pass
                    speed = np.linalg.norm(o['ball_direction'][:2])
                    efficiency = speed * o['ball'][2]  # Consider the vertical component as a factor for efficiency

                    # The reward scales with the efficiency of the high pass
                    components['high_pass_precision'][rew_index] = efficiency * 0.1  # Reward coefficient
                    reward[rew_index] += components['high_pass_precision'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
