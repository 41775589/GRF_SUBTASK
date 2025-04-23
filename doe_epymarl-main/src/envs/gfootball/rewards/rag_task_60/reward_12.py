import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for enhancing defensive adaptation through precise 
       and coordinated stopping and starting movements."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for serialization."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from deserialization."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Calculate augmented reward function that promotes defensive adaptation."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_adaptation_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate if the player is starting or stopping abruptly
            movement_magnitude = np.linalg.norm(o['left_team_direction'][o['active']])
            if movement_magnitude < 0.01 or movement_magnitude > 0.5:
                # Heavily weight appropriate starts/stops detected
                components["defensive_adaptation_reward"][rew_index] = 0.2

            reward[rew_index] += components["base_score_reward"][rew_index] + \
                                 components["defensive_adaptation_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Execute environment step and augment reward calculation."""
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
