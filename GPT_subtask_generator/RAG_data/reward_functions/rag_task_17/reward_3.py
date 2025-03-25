import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that improves reinforcement learning by incentivizing positioning and high pass movements to
    stretch the opponent's defense and create space.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the sticky actions counter and return the environment's reset observation."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return the state of the underlying environment."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the underlying environment."""
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Modifies the reward by including additional metrics to promote wide field plays and high passing
        behaviors while in possession of the ball, aimed at stretching the opponent's defense and creating space.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['left_team'][o['active']] if o['active'] < len(o['left_team']) else o['right_team'][o['active']]
            
            # Encourage wide field movements: check if player is near the sidelines
            if abs(active_player_pos[1]) > 0.3:
                components["positional_reward"][rew_index] = 0.05  # Moderate reward for being at the edges
            
            # Encourage successful high passes in the forward direction
            if o['sticky_actions'][9] == 1 and o['ball_direction'][1] > 0:
                components["high_pass_reward"][rew_index] = 0.1  # Reward for performing high passes forward
            
            # Sum up the components to determine the total reward
            reward[rew_index] = (1 * components["base_score_reward"][rew_index] +
                                 components["positional_reward"][rew_index] +
                                 components["high_pass_reward"][rew_index])

        return reward, components

    def step(self, action):
        """Perform a step using the given action, modify the reward, and return the step information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
