import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing offensive strategies like shooting, dribbling, and accurate passing."""

    def __init__(self, env):
        super().__init__(env)
        self._collected_shoot_points = {}
        self.shooting_reward = 0.1
        self.passing_reward = 0.05
        self.dribbling_reward = 0.025
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._collected_shoot_points = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_shoot_points
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_shoot_points = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        if not observation:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage shooting when close to the goal
            if o['ball_owned_team'] == 0 and o['ball'][0] > 0.7:
                if self._collected_shoot_points.get(rew_index, 0) == 0:
                    components["shooting_reward"][rew_index] = self.shooting_reward
                    reward[rew_index] += components["shooting_reward"][rew_index]
                    self._collected_shoot_points[rew_index] = 1

            # Encourage passing, especially creative ones in the offensive third
            if o['sticky_actions'][6] == 1 or o['sticky_actions'][5] == 1:  # action=long_pass or high_pass
                if o['ball_owned_team'] == 0 and o['ball'][0] > 0.3:
                    components["passing_reward"][rew_index] = self.passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]

            # Encourage dribbling using the action dribble in offensive moves
            if o['sticky_actions'][9] == 1:  # action=dribble
                if o['ball_owned_team'] == 0:
                    components["dribbling_reward"][rew_index] = self.dribbling_reward
                    reward[rew_index] += components["dribbling_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        info["final_reward"] = sum(reward)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
