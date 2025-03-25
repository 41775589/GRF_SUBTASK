import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on defensive maneuvers and midfield control."""
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = {}
        self.midfield_control = {}

    def reset(self):
        """Resets the environment and clears any persistent variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = {}
        self.midfield_control = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state including defensive and midfield control data."""
        to_pickle['defensive_positions'] = self.defensive_positions
        to_pickle['midfield_control'] = self.midfield_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state including defensive and midfield control data."""
        from_pickle = self.env.set_state(state)
        self.defensive_positions = from_pickle['defensive_positions']
        self.midfield_control = from_pickle['midfield_control']
        return from_pickle

    def reward(self, reward):
        """Calculate rewards based on defensive effectiveness and midfield control."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "defensive_reward": [0.0] * len(reward), 
                      "midfield_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, obs in enumerate(observation):
            # Defensive reward based on maintaining position and intercepting passes
            if obs['game_mode'] in (2, 4, 5) and obs['ball_owned_team'] == 0:
                components["defensive_reward"][i] = 0.2 # simulate effective defense
                reward[i] += components["defensive_reward"][i]

            # Midfield control - effective transitions and ball handling in middle of field
            if abs(obs['ball'][0]) < 0.5 and obs['ball_owned_team'] == 0:
                components["midfield_reward"][i] = 0.1 # assume effective midfield gameplay
                reward[i] += components["midfield_reward"][i]

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
