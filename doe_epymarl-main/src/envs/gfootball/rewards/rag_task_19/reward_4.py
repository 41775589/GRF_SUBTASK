import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards focused on defense and midfield control."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define internal state management
        self.defensive_actions = {}
        self.midfield_control = {}
        self.defensive_coefficient = 0.5
        self.midfield_coefficient = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions = {}
        self.midfield_control = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        state = {'defensive_actions': self.defensive_actions,
                 'midfield_control': self.midfield_control}
        return self.env.get_state({**to_pickle, **state})

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_actions = from_pickle.get('defensive_actions', {})
        self.midfield_control = from_pickle.get('midfield_control', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "midfield_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            if 'ball_owned_team' in obs:
                # Defensive reward: Increase for ball clearances and tackles
                if obs['game_mode'] in [3, 4, 5] and obs['ball_owned_team'] == 0:
                    # Reward defensive clearances and set pieces defending
                    components['defensive_reward'][index] = self.defensive_coefficient
                    reward[index] += components['defensive_reward'][index]

                # Midfield control rewards: passing and maintaining possession in midfield
                if 0.3 <= obs['ball'][0] <= 0.7 and obs['ball_owned_team'] == 0:
                    components['midfield_reward'][index] = self.midfield_coefficient
                    reward[index] += components['midfield_reward'][index]

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
