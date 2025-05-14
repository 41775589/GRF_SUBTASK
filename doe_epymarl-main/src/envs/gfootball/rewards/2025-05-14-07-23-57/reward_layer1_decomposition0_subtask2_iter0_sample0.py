import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on positional control and passing for mastery in integrating midfield and defensive play."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_thresholds = np.linspace(-1, 1, num=11)  # Thresholds to encourage good positioning
        self.last_ball_position = None

    def reset(self):
        """
        Reset the environment state and prepare for new episode. Reset the positional counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Package the current state of the environment for saving.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the environment state based on the loaded state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Compute the custom reward, focusing on ball possession improvements and passing efficiency between defense and midfield.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positional_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_position = o['left_team'][o['active']][:2]  # Get active player (x, y) position assuming it's the left team
            
            # Reward for maintaining ball possession
            if self.last_ball_position is not None and o['ball_owned_team'] == 0:  # Assuming our team is '0'
                ball_distance_moved = np.linalg.norm(self.last_ball_position - o['ball'][:2])
                components["positional_control_reward"][rew_index] = ball_distance_moved
            
            # Encourage passing by comparing between current and last possession
            if o['ball_owned_team'] == 0 and self.last_ball_position is not None:
                ball_change_of_possession = np.linalg.norm(current_position - self.last_ball_position)
                components["positional_control_reward"][rew_index] += 5.0 * np.exp(-ball_change_of_possession)  # Exponential decay to reward closer passes
            
            reward[rew_index] += components["positional_control_reward"][rew_index]
            self.last_ball_position = o['ball'][:2]

        return reward, components

    def step(self, action):
        """
        Perform a step using the given action, reward transformations are applied in this step.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
