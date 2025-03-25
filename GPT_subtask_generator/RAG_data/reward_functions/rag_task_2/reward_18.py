import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward to encourage defensive strategies and collaboration."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._team_defense_reinforcement = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.__dict__
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.__dict__ = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_coordination_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if self._is_effective_defense(o):
                components["defense_coordination_reward"][rew_index] = self._team_defense_reinforcement
                reward[rew_index] += components["defense_coordination_reward"][rew_index]

        return reward, components

    def _is_effective_defense(self, observation):
        # Basic criteria for effective defense:
        # 1) Ball owned by opponent team
        # 2) Close defenders count
        ball_owned_team = observation.get('ball_owned_team', 1)
        if ball_owned_team != 1:
            return False  # Ball is not with opponent

        own_team_pos = observation.get('left_team', []) if ball_owned_team == 1 else observation.get('right_team', [])
        opponent_pos = observation.get('right_team', []) if ball_owned_team == 1 else observation.get('left_team', [])
        ball_pos = observation.get('ball', [0, 0])

        close_defenders_count = 0
        for pos in own_team_pos:
            if np.linalg.norm(np.array(pos) - np.array(ball_pos)) < 0.05:  # Threshold for "close" defenders
                close_defenders_count += 1

        # Reward condition: more than 2 players in close defense towards ball
        return close_defenders_count >= 2

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
