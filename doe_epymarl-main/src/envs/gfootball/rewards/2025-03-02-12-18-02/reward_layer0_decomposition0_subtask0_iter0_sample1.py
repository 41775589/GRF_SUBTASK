import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for passing actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._collected_checkpoints = {0: 0, 1: 0, 2: 0}  # Each agent starts with 0 checkpoints
        self._num_checkpoints = 10  # Maximum number of checkpoints
        self._checkpoint_reward = 0.1  # Reward for each checkpoint

    def reset(self):
        self._collected_checkpoints = {0: 0, 1: 0, 2: 0}  # Reset collected checkpoints to 0 for each agent
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            agent_id = o['active']

            # Check the agent's action for passing
            if 'action' in o and o['action'] in ['Short Pass', 'High Pass', 'Long Pass']:
                # Check distance to the opponent's goal for a checkpoint
                if agent_id in self._collected_checkpoints:
                    d_to_goal = self._distance_to_opponent_goal(o['left_team'][agent_id])
                    if d_to_goal < 0.2 and self._collected_checkpoints[agent_id] < self._num_checkpoints:
                        components["checkpoint_reward"][agent_id] = self._checkpoint_reward
                        reward[rew_index] += self._checkpoint_reward
                        self._collected_checkpoints[agent_id] += 1

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def _distance_to_opponent_goal(self, player_position):
        # Calculate the distance from player position to the opponent's goal
        opponent_goal_position = [1, 0]
        distance = np.linalg.norm(np.array(player_position) - np.array(opponent_goal_position))
        return distance
