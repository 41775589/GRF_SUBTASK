import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense midfield dynamics reward. Focuses on central and wide midfield roles."""

    def __init__(self, env):
        super().__init__(env)
        self._possession_achievements = {}
        self._position_rewards = {
            "central": 0.05,  # increased influence centrally for transitions and control
            "wide": 0.03       # reward for operations on the periphery aiding wide transitions
        }
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._possession_achievements = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._possession_achievements
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._possession_achievements = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_play_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                player_y = o['right_team'][o['active']][1]
                reward_type = "central" if abs(player_y) <= 0.1 else "wide"
                player_rwd_key = (idx, reward_type)

                if player_rwd_key not in self._possession_achievements:
                    components["midfield_play_reward"][idx] = self._position_rewards[reward_type]
                    reward[idx] += components["midfield_play_reward"][idx]
                    self._possession_achievements[player_rwd_key] = True

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
