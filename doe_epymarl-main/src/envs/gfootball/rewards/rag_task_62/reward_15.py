import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward related to shooting techniques and pressure scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._checkpoint_distances = np.linspace(0.8, 0.6, 5)  # Defining 5 zones closer to the goal
        self._zone_rewards = np.linspace(0.2, 1, 5)  # Incremental rewards for each zone closer to goal
        self._pressure_multiplier = 1.5  # Reward multiplier under pressure
        self._collected_rewards = [False] * len(self._checkpoint_distances)

    def reset(self):
        self._collected_rewards = [False] * len(self._checkpoint_distances)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Compute dense rewards for shooting techniques and pressure scenarios."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_technique_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'ball' not in o:
                continue
            
            ball_pos = o['ball']
            goal_distance = np.abs(ball_pos[0] - 1)  # Distance from the right goal on x axis

            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'active' in o:
                # Calculate dynamic rewards based on proximity to the opponent's goal and pressure near goal
                for idx, threshold in enumerate(self._checkpoint_distances):
                    if goal_distance < threshold  and not self._collected_rewards[idx]:
                        # Higher reward for closer quadrants and when under pressure
                        under_pressure = len([p for p in o['right_team'] if np.linalg.norm(p - ball_pos[:2]) < 0.1]) > 2
                        multiplier = self._pressure_multiplier if under_pressure else 1.0
                        components["shooting_technique_reward"][rew_index] += self._zone_rewards[idx] * multiplier
                        self._collected_rewards[idx] = True
        
        for i in range(len(reward)):
            reward[i] += components["shooting_technique_reward"][i]

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
