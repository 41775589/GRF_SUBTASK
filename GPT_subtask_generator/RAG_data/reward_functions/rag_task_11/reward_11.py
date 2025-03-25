import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive plays and precision movements."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._checkpoint_distance = 0.1  # distance thresholds for offensive plays
        self._possession_bonus = 0.05     # bonus for keeping possession in offensive half

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
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "possession_bonus": [0.0] * len(reward),
                      "checkpoint_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x, ball_y = o['ball'][0], o['ball'][1]

            # Reward for moving towards the opponent's goal
            if o['ball_owned_team'] == 0:  # if the left team owns the ball
                # Progressive reward based on ball's x position, higher as it's closer to the opponent's goal
                reward[rew_index] += max(0.0, (ball_x + 1) / 2) * self._checkpoint_distance

                # Possession bonus for maintaining possession in opponent's half
                if ball_x > 0:
                    components["possession_bonus"][rew_index] = self._possession_bonus
                    reward[rew_index] += components["possession_bonus"][rew_index]

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
        return observation, reward, done, info
