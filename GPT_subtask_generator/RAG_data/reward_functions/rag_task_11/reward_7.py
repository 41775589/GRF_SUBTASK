import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a goal-based reward with emphasis on perfect ball control and rapid attacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._checkpoint_positions = [(0.2 * i, -0.6 + 0.2 * j) for i in range(5) for j in range(5)]
        self.collected_checkpoints = set()
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), "control_accuracy_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage controlling the ball specifically in high pressure/speed scenarios
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                for checkpoint in self._checkpoint_positions:
                    if np.linalg.norm(np.array(checkpoint) - o['ball'][:2]) < 0.1 and checkpoint not in self.collected_checkpoints:
                        components["control_accuracy_reward"][rew_index] = 0.05
                        self.collected_checkpoints.add(checkpoint)
                        break
            
            # Additional reward for quick handling and dribbling under pressure
            if o['sticky_actions'][9]:  # Dribbling
                components["control_accuracy_reward"][rew_index] += 0.01
            
            # Effective maneuvering reward
            player_pos = o['left_team'][o['active']]
            if np.linalg.norm(np.array([1, 0]) - player_pos) < 0.2:  # close to scoring area
                components["control_accuracy_reward"][rew_index] += 0.02

            reward[rew_index] += sum(components.values())

        return reward, components

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_checkpoints = set()
        return self.env.reset()

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

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = list(self.collected_checkpoints)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_checkpoints = set(from_pickle['CheckpointRewardWrapper'])
        return from_pickle
