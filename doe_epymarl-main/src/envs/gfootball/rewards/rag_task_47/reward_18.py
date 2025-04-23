import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes mastering sliding tackles near the defensive third during counter-attacks and high-pressure situations."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters for custom reward logic
        self._sliding_tackle_reward = 1.0
        self.viable_zones = []
        self.setup_zones()

    def reset(self):
        """Reset the wrapper state to initial state."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Get the current state of the wrapper."""
        to_pickle['CheckpointRewardWrapper'] = self.viable_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper from a stored state."""
        from_pickle = self.env.set_state(state)
        self.viable_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def setup_zones(self):
        """Sets up zones near the defensive third for rewarding sliding tackles."""
        border_of_defensive_third = -1/3  # the x-coordinate border delineating the defensive third of the field
        self.viable_zones = [border_of_defensive_third, -1]  # covering from border till the goal line

    def reward(self, reward):
        """Custom reward logic based on sliding tackles in specific zones during high pressure."""
        components = {"base_score_reward": reward.copy(), "tackle_reward": [0.0] * len(reward)}
        # Retrieve observations to check positions and events
        observation = self.env.unwrapped.observation()
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x_position = o['ball'][0]
            
            # Check if the ball is in the defensive third
            if self.viable_zones[0] <= ball_x_position <= self.viable_zones[1]:
                # Check if a sliding tackle sticky action is active
                # Sticky action index for sliding can vary, assumed it's 7 or similar for this example
                if o['sticky_actions'][7] == 1:
                    # Increase the reward for effective sliding tackle in the defensive third
                    components["tackle_reward"][rew_index] = self._sliding_tackle_reward
                    # Modify the corresponding reward component
                    reward[rew_index] += components["tackle_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Processing steps with custom reward modifications."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
