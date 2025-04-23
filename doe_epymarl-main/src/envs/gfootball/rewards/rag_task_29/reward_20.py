import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dense precision shot rewards near the opponent goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_accuracy_bonus = 0.2
        self.close_range_thresholds = [0.04, 0.08]  # Two close-range thresholds in y-axis near goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_shot_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and abs(o['ball'][1]) < self.close_range_thresholds[0]:
                # More bonus when closer and aligned to goal
                components["precision_shot_reward"][rew_index] = self.shot_accuracy_bonus * 2
            elif o['ball_owned_team'] == 1 and abs(o['ball'][1]) < self.close_range_thresholds[1]:
                components["precision_shot_reward"][rew_index] = self.shot_accuracy_bonus
            reward[rew_index] += components["precision_shot_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Collect stats for reward components
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in self.env.unwrapped.observation():
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
