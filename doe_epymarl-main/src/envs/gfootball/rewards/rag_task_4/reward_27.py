import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances dribbling and sprint skills in football."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize tracking the dribbles and sprints
        self.dribble_track = {}
        self.sprint_track = {}
        self.dribble_reward = 0.05  # Reward for successful dribble
        self.sprint_reward = 0.03   # Reward for employing sprint
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Needed as per the given setup

    def reset(self):
        self.dribble_track = {}
        self.sprint_track = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'dribble_track': self.dribble_track,
            'sprint_track': self.sprint_track
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        data = from_pickle['CheckpointRewardWrapper']
        self.dribble_track = data['dribble_track']
        self.sprint_track = data['sprint_track']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for agent_idx, obs in enumerate(observation):
            # Check if agent has performed a dribble
            if obs['sticky_actions'][9] == 1 and self.dribble_track.get(agent_idx, 0) == 0:
                components["dribble_reward"][agent_idx] = self.dribble_reward
                reward[agent_idx] += components["dribble_reward"][agent_idx]
                self.dribble_track[agent_idx] = 1  # Mark dribble as tracked
            
            # Check if the agent is sprinting
            if obs['sticky_actions'][8] == 1 and self.sprint_track.get(agent_idx, 0) == 0:
                components["sprint_reward"][agent_idx] = self.sprint_reward
                reward[agent_idx] += components["sprint_reward"][agent_idx]
                self.sprint_track[agent_idx] = 1  # Mark sprint as tracked

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
