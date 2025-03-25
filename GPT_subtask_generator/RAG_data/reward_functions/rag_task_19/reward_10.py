import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive and midfield tactical achievements."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To keep track of sticky actions
        
        # Configuration for midfield control and defensive stances
        self.midfield_control_reward = 0.03
        self.defensive_stance_reward = 0.05
        self.aggressive_defense_position = -0.5  # Threshold for aggressive defense

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control": [0.0] * len(reward),
                      "defensive_stance": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x = o['ball'][0]

            # Reward for controlling the midfield
            if abs(ball_x) < 0.2:  # Midfield zone
                components["midfield_control"][rew_index] = self.midfield_control_reward
                reward[rew_index] += components["midfield_control"][rew_index]

            # Reward for maintaining a defensive stance
            defensive_players = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
            # Check if any defensive player is positioned behind the aggressive_defense_position
            if any(p[0] < self.aggressive_defense_position for p in defensive_players):
                components["defensive_stance"][rew_index] = self.defensive_stance_reward
                reward[rew_index] += components["defensive_stance"][rew_index]

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
