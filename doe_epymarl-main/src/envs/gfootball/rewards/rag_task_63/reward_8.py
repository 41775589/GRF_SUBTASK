import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward function tailored for training a goalkeeper."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize custom attributes for goalkeeper training
        self._ball_saves = 0
        self._distribution_decisions = 0
        self._communications = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_saves = 0
        self._distribution_decisions = 0
        self._communications = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = {
            'ball_saves': self._ball_saves,
            'distribution_decisions': self._distribution_decisions,
            'communications': self._communications
        }
        to_pickle['CheckpointRewardWrapper'] = state_info
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle.get('CheckpointRewardWrapper', {})
        self._ball_saves = state_info.get('ball_saves', 0)
        self._distribution_decisions = state_info.get('distribution_decisions', 0)
        self._communications = state_info.get('communications', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeping_reward": [0.0] * len(reward)}

        # Assuming observation and reward list length matches the number of agents
        for i in range(len(reward)):
            o = observation[i]

            # Reward for saving goals: increase when the goalkeeper stops the ball
            if o['game_mode'] == 6 and o['ball_owned_team'] == 0:  # Penalty save scenario
                components['goalkeeping_reward'][i] += 1.0
                self._ball_saves += 1

            # Reward for quick and accurate distribution under pressure
            if o['game_mode'] == 0 and o['ball_owned_team'] == 0 and np.linalg.norm(o['ball_direction']) > 0.1:
                components['goalkeeping_reward'][i] += 0.5
                self._distribution_decisions += 1

            # Reward for positive communications/actions with defenders
            # Hypothetical communication measurement
            if o['left_team_direction'][o['active']].dot(o['ball_direction']) > 0:
                components['goalkeeping_reward'][i] += 0.2
                self._communications += 1

            # Combine rewards
            reward[i] += components['goalkeeping_reward'][i]

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
