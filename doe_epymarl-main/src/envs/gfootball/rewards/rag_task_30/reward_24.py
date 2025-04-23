import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering strategic positioning, lateral and backward movement, and accelerating 
    the transition from defense to attack."""
    
    def __init__(self, env):
        super().__init__(env)
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
        components = {"base_score_reward": reward.copy(),
                      "strategic_positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage lateral movements
            if o['sticky_actions'][1] or o['sticky_actions'][7]:  # action_top_left or action_bottom_left
                components["strategic_positioning_reward"][rew_index] += 0.01
            if o['sticky_actions'][3] or o['sticky_actions'][5]:  # action_top_right or action_bottom_right
                components["strategic_positioning_reward"][rew_index] += 0.01
            
            # Reward backward movements to promote retreating actions for defense setup
            if o['sticky_actions'][6]:  # action_bottom
                components["strategic_positioning_reward"][rew_index] += 0.02
            
            # Reward switching from defense to attack
            if (o['ball_owned_team'] == 0 and o['left_team_active'].all()) or \
                    (o['ball_owned_team'] == 1 and o['right_team_active'].all()):
                components["strategic_positioning_reward"][rew_index] += 0.05

            reward[rew_index] += components["strategic_positioning_reward"][rew_index]
        
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
