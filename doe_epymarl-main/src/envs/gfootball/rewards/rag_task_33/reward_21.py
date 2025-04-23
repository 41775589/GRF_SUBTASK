import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful long-range shots and controlling the game outside the penalty box."""

    def __init__(self, env):
        super().__init__(env)
        self.long_shot_distance_threshold = 0.7  # Approximately 30% of the field away from the goal
        self.reward_for_long_shot = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_shot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x_position = o['ball'][0]
            goal_position = 1 if o['ball_owned_team'] == 0 else -1

            # Long shot goal condition
            if ball_x_position * goal_position > self.long_shot_distance_threshold:
                if o['ball_owned_team'] == o['ball_owned_player'] and o['score'][rew_index] > 0:
                    components["long_shot_reward"][rew_index] = self.reward_for_long_shot
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
