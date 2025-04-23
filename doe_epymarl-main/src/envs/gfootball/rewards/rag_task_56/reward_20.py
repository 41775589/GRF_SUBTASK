import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward function to focus on defensive capabilities of goalkeepers and defenders."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Keeping track of defensive actions and their weights in the reward system
        self.goalkeeper_actions_weight = 1.5
        self.defender_actions_weight = 1.2
        self.basic_reward_scale = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_bonus": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Goalkeeper specific handling
            if o['active'] and o['left_team_roles'][o['active']] == 0 and o['ball_owned_team'] == 0:
                # Goalkeeper bonus for possession
                components["defensive_bonus"][rew_index] += self.goalkeeper_actions_weight
                reward[rew_index] += components["defensive_bonus"][rew_index]
                
            # Defender specific handling
            if o['active'] and o['left_team_roles'][o['active']] in [1, 2, 3, 4] and o['ball_owned_team'] == 0:
                # Defender bonus for possession
                components["defensive_bonus"][rew_index] += self.defender_actions_weight
                reward[rew_index] += components["defensive_bonus"][rew_index]
                
            # General reward scaling
            reward[rew_index] *= self.basic_reward_scale

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
