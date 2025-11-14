import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for dispossessing maneuvers in defensive areas."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = self.env.get_state(to_pickle)
        state_info['sticky_actions_counter'] = self.sticky_actions_counter
        return state_info

    def set_state(self, state):
        state_info = self.env.set_state(state)
        self.sticky_actions_counter = state_info.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return state_info

    def reward(self, reward):
        # Access observations from the environment
        observations = self.env.unwrapped.observation()
        
        # Initialize reward computation elements
        components = {
            "base_score_reward": reward.copy(),
            "dispossession_reward": [0.0] * len(reward)
        }

        if observations is None:
            return reward, components

        for idx, obs in enumerate(observations):
            # Assuming that the `sticky_actions` array has stops and slide tackles at specific indexes:
            # e.g., index 6 for Stop-Moving, index 7 for Sliding.
            stop_action = obs['sticky_actions'][6]
            slide_action = obs['sticky_actions'][7]

            posX = obs['left_team'][obs['active']][0]  # Get X position of the active player
            is_in_defensive_area = posX < -0.5  # Define defensive area on left half from center

            # Check if the controlled player performed stopping or sliding tackles in the defensive area
            if (stop_action or slide_action) and is_in_defensive_area:
                components["dispossession_reward"][idx] += 1.0  # Grant a unit reward

        for idx in range(len(reward)):
            # Calculate final reward for each agent
            reward[idx] += components["dispossession_reward"][idx]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_active
        return observation, reward, done, info
