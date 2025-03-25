import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive actions to prevent scoring."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._intercept_reward = 0.3  # Reward for intercepting the ball
        self._block_reward = 0.2  # Reward for blocking an opponent's movement
        self._tackle_reward = 0.5  # Reward for a successful tackle

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_rewards'] = {
            'intercept': self._intercept_reward,
            'block': self._block_reward,
            'tackle': self._tackle_reward
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        rewards = from_pickle['defensive_rewards']
        self._intercept_reward = rewards['intercept']
        self._block_reward = rewards['block']
        self._tackle_reward = rewards['tackle']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_rewards": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Dynamic rewards based on defensive actions
            defensive_actions = 0.0

            # Check for interceptions and blocks
            if o['game_mode'] in [0, 2, 4]:  # Normal, GoalKick, Corner
                if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                    # Assuming the agent's team is 0, and opponent is 1
                    defensive_actions += self._intercept_reward

            # Check for successful tackles
            if 'sticky_actions' in o and o['sticky_actions'][9]:  # Tackle action index
                defensive_actions += self._tackle_reward

            components["defensive_rewards"][rew_index] += defensive_actions
            reward[rew_index] += components["defensive_rewards"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
