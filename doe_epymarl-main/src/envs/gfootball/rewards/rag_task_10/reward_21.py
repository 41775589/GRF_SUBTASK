import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on defensive skills like positioning, interception, marking, and tackling."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the defensive actions that we want to reward
        self.defensive_actions = {
            'slide': 10,  # Suppose the index for action_slide is 10
            'intercept': 11,  # Suppose the index for action_intercept is 11
            'marking': 12,  # Suppose the index for action_marking is 12
            'tackling': 13  # Suppose the index for action_tackling is 13
        }
        self.defensive_multiplier = 0.05  # Reward weight for defensive actions

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
                      "defensive_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            stickies = obs['sticky_actions']
            for idx, act in self.defensive_actions.items():
                if stickies[act]:
                    components["defensive_reward"][rew_index] += self.defensive_multiplier
            reward[rew_index] += components["defensive_reward"][rew_index]
        
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
