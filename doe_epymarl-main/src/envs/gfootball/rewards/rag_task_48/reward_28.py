import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes from midfield aimed at creating scoring opportunities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_accuracy_reward = 0.2

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
                      "high_pass_accuracy_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Parse relevant information from observations
            midfield_zone = -0.3 < o['ball'][0] < 0.3  # Approximation of midfield range on the x-axis
            ball_in_air = o['ball'][2] > 0.1  # Checking height condition assuming non-trivial z value signifies air-borne
            active_player = o['active']
            # Condition for rewarding accurate high pass from midfield
            if midfield_zone and o['ball_owned_team'] == 0 and o['ball_owned_player'] == active_player:
                if ball_in_air and o['sticky_actions'][9]:  # Assuming index 9 corresponds to a high pass in sticky actions
                    components['high_pass_accuracy_reward'][rew_index] = self.high_pass_accuracy_reward
                    reward[rew_index] += 1.5 * components['high_pass_accuracy_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
