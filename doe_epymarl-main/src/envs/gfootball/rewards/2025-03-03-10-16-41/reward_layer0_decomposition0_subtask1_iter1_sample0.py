import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive actions."""

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
        components = {"base_score_reward": reward.copy(), "checkpoint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Defensive actions like Sliding, Stop-Dribble, Stop-Sprint in the second group
            if rew_index in [agent_index_1, agent_index_2]: # Replace agent_index_1 and agent_index_2 with actual agent indices in the group
                if o['action'] == 'Sliding' or o['action'] == 'Stop-Dribble' or o['action'] == 'Stop-Sprint':
                    # If defensive action is taken
                    components["checkpoint_reward"][rew_index] = self._checkpoint_reward * (self._num_checkpoints - self._collected_checkpoints.get(rew_index, 0))
                    reward[rew_index] += self._checkpoint_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
