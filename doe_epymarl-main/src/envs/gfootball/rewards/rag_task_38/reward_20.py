import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on effectiveness in counterattacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._long_pass_threshold = 0.5  # threshold to consider a pass as 'long'
        self._transition_coef = 1.0       # coefficient for transitions effectiveness
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(),   # keep the original score reward
                      "counterattack_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for long passes when transitioning from defense to attack
            did_pass = o['ball_direction'][0] > 0 and np.linalg.norm(o['ball_direction'][:2]) > self._long_pass_threshold
            in_defensive_third = o['ball'][0] < -0.33 and o['ball_owned_team'] == 0

            if did_pass and in_defensive_third:
                components["counterattack_reward"][rew_index] = self._transition_coef

            # Aggregate the rewards
            reward[rew_index] += components["counterattack_reward"][rew_index]
            
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
