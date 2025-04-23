import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for mastering wide midfield responsibilities,
    focusing on accurate High Pass usage and effective lateral positioning.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize action-related counters and game-related state
        self.high_pass_usage = [0, 0]  # counts High Pass actions made by agents
        self.last_positions = [None, None]  # previous positions of the ball
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # counts current sticky actions
        
    def reset(self):
        # Reset counters and state upon environment reset
        self.high_pass_usage = [0, 0]
        self.last_positions = [None, None]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Store state for resumption
        to_pickle['CheckpointRewardWrapper'] = {'high_pass_usage': self.high_pass_usage,
                                                'last_positions': self.last_positions}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Resume from stored state
        from_pickle = self.env.set_state(state)
        self.high_pass_usage = from_pickle['CheckpointRewardWrapper']['high_pass_usage']
        self.last_positions = from_pickle['CheckpointRewardWrapper']['last_positions']
        return from_pickle

    def reward(self, reward):
        # Update reward based on wide midfield actions and positioning
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'high_pass_reward': [0.0, 0.0],
                      'positioning_reward': [0.0, 0.0]}
                      
        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            # Detect high pass action
            if o['sticky_actions'][9] == 1:  # assuming index 9 is High Pass
                self.high_pass_usage[idx] += 1
                components['high_pass_reward'][idx] = 0.05  # reward for using High Pass
            
            # Reward for maintaining position at wide areas
            if 'right_team' in o:
                # Check if agent is one of the wide midfielders, typically indices 6 and 7
                if o['active'] in [6, 7]:
                    x_pos = o['right_team'][o['active']][0]
                    y_pos_abs = abs(o['right_team'][o['active']][1])
                    # Check if the player is in a wide midfield area
                    if y_pos_abs > 0.2:
                        positioning_quality = y_pos_abs / 0.42  # normalize by field width
                        components['positioning_reward'][idx] = 0.1 * positioning_quality
            
            # Update reward calculation
            reward[idx] += components['high_pass_reward'][idx] + components['positioning_reward'][idx]

        return reward, components

    def step(self, action):
        # Execute a step in the environment, compute the reward, and return modified observation and info
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
