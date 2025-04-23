import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards offensive actions like accurate shooting, dribbling, and various types of passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_reward = 0.2
        self.pass_reward = 0.1
        self.dribble_reward = 0.05

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
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        components.update({"shooting_reward": [0.0] * len(reward),
                           "pass_reward": [0.0] * len(reward),
                           "dribble_reward": [0.0] * len(reward)})

        for rew_index, o in enumerate(observation):
            if o['game_mode'] == 6:  # Check if in Shooting mode
                components["shooting_reward"][rew_index] = self.shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]
            
            if 'sticky_actions' in o:
                # Long and high passes typically require a combination of right-side actions and a pass
                if o['sticky_actions'][4] and o['sticky_actions'][9]:  # Action right (4) and Action dribble (9)
                    components["pass_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += components["pass_reward"][rew_index]
                # Dribbling, checking for dribble action active
                elif o['sticky_actions'][9]:  # Action dribble
                    components["dribble_reward"][rew_index] = self.dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track actions used for potential later use or analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                # Increment action counts for sticky actions
                self.sticky_actions_counter[i] += (action > 0)

        return observation, reward, done, info
