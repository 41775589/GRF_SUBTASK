import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward to incentivize players to move closer to the opponent's goal."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._collected_checkpoints = {} 
        self._num_checkpoints = 10
        self._checkpoint_reward = 0.1

    def reset(self):
        self._collected_checkpoints = {}
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
            if reward[rew_index] == 1:
                components["checkpoint_reward"][rew_index] = self._checkpoint_reward * (
                        self._num_checkpoints -
                        self._collected_checkpoints.get(rew_index, 0))
                reward[rew_index] = 1 * components["base_score_reward"][rew_index] + components["checkpoint_reward"][
                    rew_index]
                self._collected_checkpoints[rew_index] = self._num_checkpoints
                continue

            opponent_goal_position = [1.0, 0.0]  # Define the opponent's goal position
            player_position = o['left_team'][play_index]  # Get the player's position
            distance_to_goal = np.linalg.norm(np.array(player_position) - np.array(opponent_goal_position))
            
            # Collect the checkpoints.
            # We give reward for reducing the distance from 1.0 to 0.2.
            while (self._collected_checkpoints.get(rew_index, 0) < self._num_checkpoints):
                if distance_to_goal > 0.2:
                    break
                components["checkpoint_reward"][rew_index] = self._checkpoint_reward
                reward[rew_index] += 1.5 * components["checkpoint_reward"][rew_index]
                self._collected_checkpoints[rew_index] = (
                        self._collected_checkpoints.get(rew_index, 0) + 1)
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
