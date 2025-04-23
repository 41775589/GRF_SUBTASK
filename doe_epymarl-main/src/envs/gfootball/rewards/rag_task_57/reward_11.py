import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that defines a complex reward function specific to teaching agents cooperative offensive
    football strategies, particularly focusing on midfielders creating space and delivering the ball,
    and strikers finishing plays.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_coordination_reward = 0.05  # Reward for midfielders positioning.
        self.striker_finishing_reward = 0.1  # Reward for strikers finishing plays.
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
        components = {
            "base_score_reward": reward.copy(),
            "midfield_coordination_reward": [0.0] * len(reward),
            "striker_finishing_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Mechanism to reward midfield players for effective passing and spacing.
            midfielders = [index for index, role in enumerate(o['left_team_roles']) if role in [4, 5, 6]]
            if observation['active'] in midfielders:
                components["midfield_coordination_reward"][rew_index] += self.midfield_coordination_reward
            
            # Mechanism to reward strikers for effective play finishing.
            strikers = [index for index, role in enumerate(o['left_team_roles']) if role == 9]
            if observation['active'] in strikers and observation['score'][0] > observation['score'][1]:
                components["striker_finishing_reward"][rew_index] += self.striker_finishing_reward
            
            # Calculate total reward with components.
            total_reward = (components["base_score_reward"][rew_index] +
                            components["midfield_coordination_reward"][rew_index] +
                            components["striker_finishing_reward"][rew_index])

            reward[rew_index] = total_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Include debug info regarding sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
