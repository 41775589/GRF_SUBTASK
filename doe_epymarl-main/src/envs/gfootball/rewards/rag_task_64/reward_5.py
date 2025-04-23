import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for high passes and crossing strategies
       to support dynamic attacking plays."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize variables to store checkpoints for crossing the ball
        self.crossing_checkpoints = {}
        # Each checkpoint corresponds to important zones across the width of the pitch
        self.num_crossing_zones = 5
        self.crossing_reward = 0.05
        # Track the actions to penalize unnecessary actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and clear checkpoint counters."""
        self.crossing_checkpoints = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the current state with additional reward wrapper state."""
        to_pickle['CheckpointRewardWrapper'] = self.crossing_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the state and restore additional reward wrapper state."""
        from_pickle = self.env.set_state(state)
        self.crossing_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Custom reward function which rewards crossing and high passes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "crossing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if the ball is in the air, symbolizing a potential crossing/high pass
            if o['ball'][2] > 0.1:  # Using a threshold for z-axis of the ball to determine if it's in the air
                zone_index = int((o['ball'][0] + 1) / (2 / self.num_crossing_zones))
                if self.crossing_checkpoints.get(rew_index, set()).issubset({zone_index}):
                    components["crossing_reward"][rew_index] = self.crossing_reward
                    reward[rew_index] += components["crossing_reward"][rew_index]
                    self.crossing_checkpoints[rew_index].add(zone_index)

        return reward, components

    def step(self, action):
        """Steps through the environment, modifies reward, and captures additional statistics."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
