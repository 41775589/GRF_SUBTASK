import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for skills aiding in defense to attack transition."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_control_reward = 0.1
        self.dribble_reward = 0.05
        self.control_under_pressure_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_control_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "control_under_pressure_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, r in enumerate(reward):
            o = observation[rew_index]
            # Encourage maintaining ball possession with controlled, precise passes
            if o['sticky_actions'][8] == 1:  # Dribbling
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]
            if o['sticky_actions'][7]:  # Bottom left (diagonal pass/dribble approximation)
                components["pass_control_reward"][rew_index] = self.pass_control_reward
                reward[rew_index] += components["pass_control_reward"][rew_index]
            if o['ball_rotation'][2] != 0 and o['ball_owned_team'] == 0:  # Under pressure
                components["control_under_pressure_reward"][rew_index] = self.control_under_pressure_reward
                reward[rew_index] += components["control_under_pressure_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
