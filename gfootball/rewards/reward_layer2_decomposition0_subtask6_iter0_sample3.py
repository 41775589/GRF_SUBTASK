import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a sliding tackle reward for defensive maneuvers in a football game."""

    def __init__(self, env, reward_scale=1.0):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.game_mode_tracker = None
        self.last_sliding_action = False
        self.reward_scale = reward_scale

    def reset(self):
        """ Reset the game and environment states. """
        self.game_mode_tracker = None
        self.last_sliding_action = False
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Get the current state for serialization """
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {'game_mode_tracker': self.game_mode_tracker,
                                            'last_sliding_action': self.last_sliding_action}
        return state

    def set_state(self, state):
        """ Set the environment state from serialization """
        env_state = self.env.set_state(state)
        self.game_mode_tracker = env_state['CheckpointRewardWrapper']['game_mode_tracker']
        self.last_sliding_action = env_state['CheckpointRewardWrapper']['last_sliding_action']
        return env_state

    def reward(self, reward):
        """ Enhance reward function to encourage sliding tackles in defensive situations. """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        assert len(reward) == 1  # Assumption: single agent environment
        
        new_reward = reward.copy()
        components = {"base_score_reward": reward.copy(), "sliding_tackle_reward": [0.0]}

        o = observation[0]
        # Check if the game mode is in normal play
        if o['game_mode'] == 0:  # Assuming '0' denotes normal game mode
            if 'sticky_actions' in o:
                current_sliding = o['sticky_actions'][9]  # Assuming index '9' is for sliding
                if self.last_sliding_action == False and current_sliding == 1:
                    # Reward for initiating a sliding tackle
                    components["sliding_tackle_reward"][0] = 0.5 * self.reward_scale
                    new_reward += components["sliding_tackle_reward"]

                self.last_sliding_action = current_sliding == 1

        return new_reward, components

    def step(self, action):
        """ Execute an environment step with action, modify reward accordingly and return state """
        observation, reward, done, info = self.env.step(action)
        
        # Modify the reward using the reward() method
        reward, components = self.reward([reward])
        
        # Add final reward to the info
        info["final_reward"] = sum(reward)
        
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
