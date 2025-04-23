import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom wrapper to add rewards for stopping and sprinting effectively in defense scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.points = 10
        self.stop_penalty = -0.05
        self.sprint_bonus = 0.1
        self.stop_actions = [0, 6, 7]  # assuming stop actions are indexed in the environment's action set
        self.sprint_action = 8  # assuming sprint action is indexed
        self.previous_action = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the rewards."""
        self.previous_action = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment for serialization."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from deserialization."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Augment the default reward with defensive technique rewards."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "stop_sprint_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            current_action = observation[rew_index]['sticky_actions']
            current_player_position = observation[rew_index]['left_team'][observation[rew_index]['active']][:2]
            opponent_positions = observation[rew_index]['right_team']

            # Compute distance to nearest opponent
            distances = np.linalg.norm(opponent_positions - current_player_position, axis=1)
            nearest_opponent_distance = np.min(distances)

            # Check if stopped near an opponent
            if self.previous_action in self.stop_actions and nearest_opponent_distance < 0.1:
                components["stop_sprint_reward"][rew_index] += self.stop_penalty
            
            # Check for sprint action
            if current_action[self.sprint_action]:
                components["stop_sprint_reward"][rew_index] += self.sprint_bonus

            # Update the reward
            reward[rew_index] += components["stop_sprint_reward"][rew_index]

        self.previous_action = current_action
        
        return reward, components

    def step(self, action):
        """Perform a step in the environment, modifying the reward based on custom logic."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
