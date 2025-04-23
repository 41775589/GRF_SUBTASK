import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focusing on team synergy during 
    possession changes, emphasizing precise timing and strategic positioning."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._possession_change_positions = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._possession_change_positions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['possession_changes'] = self._possession_change_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._possession_change_positions = from_pickle.get('possession_changes', [])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "possession_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if reward[rew_index] != 0:
                continue

            ball_owned_team = o.get('ball_owned_team')
            ball_position = np.array(o['ball'])

            # Detect possession change
            if ball_owned_team != -1 and \
                any(o['sticky_actions'][8:10]) and \
                ball_owned_team != self._last_ball_owned_team:

                possession_reward = 0.05 * (1 - np.sum(np.abs(ball_position - np.array([0, 0]))))
                components["possession_change_reward"][rew_index] += possession_reward
                reward[rew_index] += possession_reward

                self._last_ball_owned_team = ball_owned_team
                self._possession_change_positions.append(ball_position.copy())

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
