import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on dribbling, evasion, and sprint usage in offensive scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions for dribbling and sprinting
        self.dribbling_progress = {}
        self.sprint_use_counter = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_progress = {}
        self.sprint_use_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "dribbling_progress": self.dribbling_progress,
            "sprint_use_counter": self.sprint_use_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribbling_progress = from_pickle['CheckpointRewardWrapper']['dribbling_progress']
        self.sprint_use_counter = from_pickle['CheckpointRewardWrapper']['sprint_use_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Adding rewards for dribbling skills and proper sprint usage
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Award for dribbling (ball control) progression
            if o['sticky_actions'][9]:  # Check if the dribble action is active
                components["dribbling_reward"][rew_index] = 0.05  # Increase reward when dribbling
                self.dribbling_progress[rew_index] = self.dribbling_progress.get(rew_index, 0) + 1
            if self.dribbling_progress.get(rew_index, 0) > 10:
                components["dribbling_reward"][rew_index] += 0.1  # Additional bonus for sustained dribbling

            # Award for sprinting when moving forward towards the opponent's goal
            if o['sticky_actions'][8] and o['left_team_direction'][o['active']][0] > 0:
                components["sprint_reward"][rew_index] = 0.05  # Increase reward for sprint usage in offensive movement
                self.sprint_use_counter[rew_index] = self.sprint_use_counter.get(rew_index, 0) + 1

        # Calculate the final reward for each agent
        for rew_index in range(len(reward)):
            reward[rew_index] += components["dribbling_reward"][rew_index] + components["sprint_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
