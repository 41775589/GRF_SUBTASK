import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for taking long-range shots that beat opposing defenders outside the penalty box."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.long_shot_distance_threshold = -0.6  # Near the midfield
        self._long_shot_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_shot_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encouraging long shots: Check if the ball is around midfield or closer to the opponent's goal, outside the penalty area
            if o['ball'][0] > self.long_shot_distance_threshold and o['ball_owned_team'] == o['active']:
                # Check if a shot was taken which is generally a high power kick towards the goal
                if o['game_mode'] in [0] and 'action_sprint' not in o['sticky_actions']:
                    components["long_shot_reward"][rew_index] = self._long_shot_reward
                    reward[rew_index] += components["long_shot_reward"][rew_index]

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
