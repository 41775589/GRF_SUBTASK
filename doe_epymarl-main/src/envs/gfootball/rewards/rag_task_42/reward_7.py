import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds strategy-focused rewards for mastering midfield dynamics, coordination under pressure, and strategic repositioning."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.midfield_control_counters = np.zeros(10, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_reward = 0.05
        self.pressure_handling_reward = 0.03
        self.transition_reward = 0.02

    def reset(self):
        self.midfield_control_counters.fill(0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_control_counters'] = self.midfield_control_counters
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_counters = from_pickle['midfield_control_counters']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward),
                      "pressure_handling_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Midfield control reward if any team member is in midfield area
            midfield_x_positions = [pos[0] for pos in np.concatenate([o['left_team'], o['right_team']])]
            if any(-0.2 <= x <= 0.2 for x in midfield_x_positions):
                components["midfield_control_reward"][rew_index] = self.midfield_reward
                reward[rew_index] += components["midfield_control_reward"][rew_index]
                self.midfield_control_counters[rew_index] = 1
            
            # Pressure handling reward if maintaining ball possession under opponent pressure
            if o['ball_owned_team'] == o['active'] and np.any(o['right_team_direction'] ** 2 + o['right_team_direction'] ** 2 > 0.05):
                components["pressure_handling_reward"][rew_index] = self.pressure_handling_reward
                reward[rew_index] += components["pressure_handling_reward"][rew_index]
            
            # Strategic transition reward for rapid repositioning during offensive to defensive transitions and vice versa
            if o['game_mode'] in [0, 1]:  # Normal and Kickoff modes assumed for transitions
                components["transition_reward"][rew_index] = self.transition_reward
                reward[rew_index] += components["transition_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
