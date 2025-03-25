import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards accuracy and power in shooting during various game scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._shot_accuracy_reward = 1.0
        self._shot_power_reward = 0.5
        self._loss_penalty = -0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_accuracy_reward": [0.0],
                      "shot_power_reward": [0.0]}

        if observation is None:
            return reward, components

        o = observation[0]  # assuming a single agent
        has_ball = o['ball_owned_team'] == 0

        # Reward for scoring
        components["shot_accuracy_reward"][0] = self._shot_accuracy_reward if reward[0] == 1 and has_ball else 0
        reward[0] += components["shot_accuracy_reward"][0]

        # Additional reward for high power shots (top action triggered with sprint)
        if 'sticky_actions' in o:
            is_sprinting = o['sticky_actions'][8] == 1  # sprinting index
            is_shooting = o['game_mode'] == 6  # assuming shooting correlates to game_mode=6 (penalty mode as example)
            if is_sprinting and is_shooting:
                components["shot_power_reward"][0] = self._shot_power_reward
                reward[0] += components["shot_power_reward"][0]

        # Penalty for losing the ball
        if not has_ball:
            reward[0] += self._loss_penalty

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
