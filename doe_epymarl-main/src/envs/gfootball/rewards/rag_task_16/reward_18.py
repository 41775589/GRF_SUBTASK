import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focused on enhancing skills in executing high passes effectively. The 
    reward system emphasizes control of trajectory, correct power assessment, and situational 
    application of high passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 5  # For simplification, using 5 checkpoints that reflect different phases in high pass execution
        self._checkpoint_reward = 0.2
        self._collected_checkpoints = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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

        # Calculate component rewards for controlling high pass trajectory and power
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['ball'] is not None and o['ball'][2] > 0.2: # Check if the ball is airborne enough to be considered a "high" pass
                # Calculate distance from the ball to the nearest player in the opposing team
                distances_to_opponents = [np.linalg.norm(o['ball'][:2] - opponent_pos) for opponent_pos in o['right_team']]
                min_distance = min(distances_to_opponents)

                # Reward for power and trajectory control: Closer to opponents, higher the requirement for precise control
                if min_distance < 0.3:
                    checkpoint = min(self._num_checkpoints - 1, int(min_distance / 0.06)) # Simplified checkpoint collection
                    if self._collected_checkpoints.get(rew_index, 0) <= checkpoint:
                        components["checkpoint_reward"][rew_index] = self._checkpoint_reward
                        reward[rew_index] += components["checkpoint_reward"][rew_index]
                        self._collected_checkpoints[rew_index] = checkpoint

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
