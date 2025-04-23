import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function focusing on mastering decisive close-range attacks against goalkeepers."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_close_range_effectiveness = 0.0
        self.dribble_effectiveness_scale = 0.2
        self.shot_precision_scale = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_close_range_effectiveness = 0.0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_close_range_effectiveness': self.previous_close_range_effectiveness
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_close_range_effectiveness = from_pickle['CheckpointRewardWrapper']['previous_close_range_effectiveness']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_precision_reward": [0.0] * len(reward), 
                      "dribble_effectiveness_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Reward for close range attacks
            if o['ball_owned_team'] == 1 and np.linalg.norm(np.array(o['ball'][:2]) - np.array([1, 0])) < 0.2:
                effectiveness = self.shot_precision_scale * (1 - np.linalg.norm(np.array(o['ball'][:2]) - np.array([1, 0])))
                components["shot_precision_reward"][rew_index] = effectiveness
                reward[rew_index] += effectiveness
            
            # Consider the dribbling effectiveness.
            if o['sticky_actions'][9] == 1:  # Dribble action is active
                components["dribble_effectiveness_reward"][rew_index] = self.dribble_effectiveness_scale
                reward[rew_index] += self.dribble_effectiveness_scale

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
