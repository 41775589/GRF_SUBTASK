import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds reward enhancements for wide midfield responsibilities, 
       focusing on high passes and lateral positioning to expand the play.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        sticky_actions_state = from_pickle.get('CheckpointRewardWrapper', {})
        self.sticky_actions_counter = np.array(sticky_actions_state.get('sticky_actions', []), dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "positional_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Encourage high pass plays.
            if o['sticky_actions'][5]:  # assuming index 5 corresponds to 'high_pass'
                components["high_pass_reward"][rew_index] = 0.05  # Slight reward increase for high pass
            
            # Reward aligning on the wide fields correctly
            player_x_pos = o['right_team'][o['active']][0]  # assuming active player's x position
            if abs(player_x_pos) > 0.5:  # moderating towards side play wider than halfway width
                components["positional_reward"][rew_index] = 0.03  # Reward players aligning wide
            
            reward[rew_index] += components["high_pass_reward"][rew_index] + components["positional_reward"][rew_index]
        
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
