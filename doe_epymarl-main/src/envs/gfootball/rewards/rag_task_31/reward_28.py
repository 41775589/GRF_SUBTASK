import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for tackling and aggressive defensive actions,
    promoting faster reactions to opposing attacks.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "defensive_play_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage tackling (assumed to be when game_mode corresponds to possession change caused by the defender)
            if o['game_mode'] == 2:  # Assuming the game_mode 2 corresponds to after possession change due to tackle
                components["defensive_play_reward"][rew_index] = 0.3
                reward[rew_index] += components["defensive_play_reward"][rew_index]

            # Reward aggressive defensive plays (e.g., slide tackling) if any action taken leads to gain of possession
            if 'action' in o and o['action'] == 'slide':
                components["defensive_play_reward"][rew_index] = 0.5
                if o['ball_owned_team'] == 0:  # assuming 0 is the team ID of the agent's team
                    reward[rew_index] += components["defensive_play_reward"][rew_index]

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
