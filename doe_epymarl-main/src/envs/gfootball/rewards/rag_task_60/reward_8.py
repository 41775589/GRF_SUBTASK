import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on players' defensive positions
       and their movements to efficiently stop and start in response to the offense."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize variables to track movement and stopping positions of players
        self.reset()

    def reset(self):
        # Reset defensive positioning tracking
        self.previous_positions = None
        self.current_positions = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_positions': self.previous_positions,
            'current_positions': self.current_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_positions = from_pickle['CheckpointRewardWrapper']['previous_positions']
        self.current_positions = from_pickle['CheckpointRewardWrapper']['current_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components
        
        components["defensive_positioning_reward"] = np.array([0.0, 0.0])

        # Calculate the reward based on the positions
        for i in range(2):
            current_obs = observation[i]

            # Update positions
            if self.previous_positions is None:
                self.previous_positions = current_obs['left_team']
                self.current_positions = current_obs['left_team']
            else:
                self.previous_positions = self.current_positions
                self.current_positions = current_obs['left_team']

            # Calculate positioning deltas
            if self.previous_positions is not None:
                movement_vectors = self.current_positions - self.previous_positions
                movement_magnitudes = np.linalg.norm(movement_vectors, axis=1)

                # Reward quick stops and starts: lower movement magnitudes get higher rewards
                stop_start_rewards = np.exp(-movement_magnitudes) * 0.1  # reward scaling

                # Sum rewards across all players, normalize it, and assign to the team's reward component
                components["defensive_positioning_reward"][i] = np.sum(stop_start_rewards)

        # Include the defensive component in reward list
        for i in range(len(reward)):
            reward[i] += components["defensive_positioning_reward"][i]

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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
