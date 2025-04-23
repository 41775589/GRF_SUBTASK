import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward system to enhance defending strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_reward = 0.2
        self._positioning_reward = 0.05
        self._passing_reward = 0.1

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
            "tackle_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward),
        }
        if observation is None:
            return reward, components

        for i, (o, base_rew) in enumerate(zip(observation, reward)):
            # Reward for tackling: increases when a tackle leads to gaining ball possession
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                components["tackle_reward"][i] = self._tackle_reward

            # Reward for efficient positioning: based on being in a defensive optimal position
            # This simplistic approach considers positions near the goal as optimal
            if o['left_team'][o['active']][0] < -0.5:
                components["positioning_reward"][i] = self._positioning_reward

            # Reward for pressured passing: Successfully passing the ball under pressure
            if o['game_mode'] in {2, 4}:  # Pressured situations: GoalKick or Corner
                if o['ball_owned_team'] == 0:
                    components["passing_reward"][i] = self._passing_reward

            # Aggregate customized rewards with basic game reward
            reward[i] = (base_rew +
                         components["tackle_reward"][i] +
                         components["positioning_reward"][i] +
                         components["passing_reward"][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
