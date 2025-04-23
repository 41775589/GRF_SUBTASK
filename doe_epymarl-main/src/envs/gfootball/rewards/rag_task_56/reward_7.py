import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive skills reward for goalkeepers and defenders."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards for successful defense and ball control by defenders
        self.defender_control_reward = 0.05
        self.goalkeeper_save_reward = 0.2
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o.get('ball_owned_team')
            active_role = o['left_team_roles'][o['active']]
            is_defender = active_role in [1, 2, 3, 4]  # assuming roles 1-4 are defenders
            is_goalkeeper = active_role == 0
            
            # Prevent goals or retain ball control
            if ball_owned_team == 0 and (is_defender or is_goalkeeper):
                if is_goalkeeper:
                    components["defense_reward"][rew_index] += self.goalkeeper_save_reward
                else:
                    components["defense_reward"][rew_index] += self.defender_control_reward
                
                reward[rew_index] += components["defense_reward"][rew_index]
        
        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

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
