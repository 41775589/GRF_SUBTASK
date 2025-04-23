import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing wide midfield responsibilities, particularly on high passing 
    and effective positioning to expand the gameplay laterally."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward components for specific midfield actions and positioning
        self.high_pass_reward = 0.2
        self.positioning_reward = 0.1

    def reset(self):
        """Reset the environment and the auxiliary variables for tracking sticky actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the sticky action counters along with the environment state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the state including the sticky action counters."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Custom reward function to enhance midfield performance in handling lateral transitions and high passes."""
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)

        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Reward for high pass actions
            if o['sticky_actions'][9]:  # '1' at index 9 corresponds to high pass
                reward[rew_index] += self.high_pass_reward * 2  # High pass is encouraged heavily
                components["high_pass_reward"][rew_index] = self.high_pass_reward

            # Reward for maintaining good positioning
            if o['ball_owned_team'] == 1 and -0.5 <= o['ball'][0] <= 0.5:
                # Check if the player is close to the side lines to encourage wide gameplay
                if abs(o['ball'][1]) >= 0.3:
                    reward[rew_index] += self.positioning_reward
                    components["positioning_reward"][rew_index] = self.positioning_reward

        return reward, components

    def step(self, action):
        """Perform a step in the environment, augmented by the custom reward function."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter for each sticky action that is active
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
