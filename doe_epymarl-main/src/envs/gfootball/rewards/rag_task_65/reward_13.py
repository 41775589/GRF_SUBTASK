import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on scenario-based training for shooting and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.passing_checkpoints = {}
        self.shooting_checkpoints = {}
        self.num_passing_zones = 5  # Divide the field into 5 zones for passing
        self.num_shooting_zones = 3  # Three zones approaching goal for shooting
        self.passing_reward = 0.05
        self.shooting_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_checkpoints = {}
        self.shooting_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'passing_checkpoints': self.passing_checkpoints,
            'shooting_checkpoints': self.shooting_checkpoints
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passing_checkpoints = from_pickle['CheckpointRewardWrapper']['passing_checkpoints']
        self.shooting_checkpoints = from_pickle['CheckpointRewardWrapper']['shooting_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Assuming 0 is the controlling team
                ball_pos = o['ball']
                # Simplified zone rewards for passing:
                x_zone = int((ball_pos[0] + 1) / 0.4)  # Normalize and scale to 5 zones
                if x_zone in range(self.num_passing_zones) and x_zone not in self.passing_checkpoints.get(rew_index, []):
                    components["passing_reward"][rew_index] = self.passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]
                    if rew_index not in self.passing_checkpoints:
                        self.passing_checkpoints[rew_index] = []
                    self.passing_checkpoints[rew_index].append(x_zone)

                # Simplified zone rewards for shooting approach:
                if ball_pos[0] > 0.5:  # Ball in the attack third
                    shooting_zone = int((ball_pos[0] - 0.5) / 0.25)
                    if shooting_zone in range(self.num_shooting_zones) and shooting_zone not in self.shooting_checkpoints.get(rew_index, []):
                        components["shooting_reward"][rew_index] = self.shooting_reward
                        reward[rew_index] += components["shooting_reward"][rew_index]
                        if rew_index not in self.shooting_checkpoints:
                            self.shooting_checkpoints[rew_index] = []
                        self.shooting_checkpoints[rew_index].append(shooting_zone)

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
