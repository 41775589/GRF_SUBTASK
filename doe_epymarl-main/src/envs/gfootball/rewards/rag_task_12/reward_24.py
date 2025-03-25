import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards based on midfield and defense capabilities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._passing_skills_checkpoint = 0.1  # Bonus for successful passes
        self._dribbling_skills_checkpoint = 0.05  # Bonus for maintaining ball possession under pressure
        self._positional_advancement_reward = 0.1  # Bonus for moving to strategic positions
        self._sprint_usage_checkpoint = 0.02  # Reward for effective sprinting

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
            "passing_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "positional_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Enhance reward for effective passing
            if 'action' in o and (o['action'] in ['High Pass', 'Long Pass']):
                components["passing_reward"][rew_index] = self._passing_skills_checkpoint
                reward[rew_index] += components["passing_reward"][rew_index]

            # Reward dribbling under pressure
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'pressure' in o:
                components["dribbling_reward"][rew_index] = self._dribbling_skills_checkpoint * o['pressure']
                reward[rew_index] += components["dribbling_reward"][rew_index]

            # Reward movement to strategic positions
            if 'position' in o and o['position'] in strategic_positions:
                components["positional_reward"][rew_index] = self._positional_advancement_reward
                reward[rew_index] += components["positional_reward"][rew_index]

            # Reward appropriate sprint usage
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # Assuming 8 is sprint
                components["sprint_reward"][rew_index] = self._sprint_usage_checkpoint
                reward[rew_index] += components["sprint_reward"][rew_index]

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
