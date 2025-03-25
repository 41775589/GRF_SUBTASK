import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for defensive and midfield interplay."""

    def __init__(self, env):
        super().__init__(env)
        self._defense_score = 0.5
        self._midfield_score = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Track positions and counts of ball intercepts by defense and passes in midfield
        self._defense_intercepts = 0
        self._midfield_passes = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defense_intercepts = 0
        self._midfield_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defense_intercepts'] = self._defense_intercepts
        to_pickle['midfield_passes'] = self._midfield_passes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defense_intercepts = from_pickle['defense_intercepts']
        self._midfield_passes = from_pickle['midfield_passes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_reward": [0.0] * len(reward),
                      "midfield_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for defensive plays (e.g., intercepting the ball)
            if o['game_mode'] in [2, 3, 4, 5, 6]:  # Defensive game modes like free kick, corner etc
                if o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] in [1, 2, 3, 4]:  # Defenders
                    self._defense_intercepts += 1
                    components["defense_reward"][rew_index] = self._defense_score
            
            # Reward for strategic midfield control (e.g., successful passes)
            if o['ball_direction'][0] != 0 or o['ball_direction'][1] != 0:  # Ball is moving
                if o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] in [5, 6, 7, 8]:  # Midfielders
                    self._midfield_passes += 1
                    components["midfield_reward"][rew_index] = self._midfield_score

            # Aggregate the rewards
            reward[rew_index] += components["defense_reward"][rew_index] + components["midfield_reward"][rew_index]

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
