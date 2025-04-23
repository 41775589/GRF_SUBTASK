import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling and sprinting effectively."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_dribbles = {}
        self.dribble_reward = 0.03
        self.sprint_reward = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_dribbles = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_dribbles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_dribbles = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        dribble_reward = [0.0] * len(reward)
        sprint_reward = [0.0] * len(reward)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward dribbling when the ball is owned by the player's team.
            if ('ball_owned_team' in o and
                o['ball_owned_team'] == o['active'] and
                'sticky_actions' in o and
                o['sticky_actions'][9] == 1):  # Dribble action index
                dribble_reward[rew_index] = self.dribble_reward
                self._collected_dribbles[rew_index] = self._collected_dribbles.get(rew_index, 0) + 1

            # Reward sprint when actively moving towards opponent's goal.
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # Sprint action index
                sprint_reward[rew_index] = self.sprint_reward

            reward[rew_index] += dribble_reward[rew_index] + sprint_reward[rew_index]

        components['dribble_reward'] = dribble_reward
        components['sprint_reward'] = sprint_reward
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
