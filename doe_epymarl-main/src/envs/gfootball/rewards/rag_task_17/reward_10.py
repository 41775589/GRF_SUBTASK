import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds custom rewards for mastering wide midfield responsibilities 
    like High Pass and positioning to expand the field of play."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to customize
        self.high_pass_reward = 0.5
        self.position_reward = 0.1
        self.high_pass_action = 9 # Assuming 9 corresponds to high pass action.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        res = self.env.set_state(state)
        from_pickle = res['sticky_actions_counter']
        self.sticky_actions_counter = from_pickle
        return res

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for using high pass.
            if o['sticky_actions'][self.high_pass_action]:
                components["high_pass_reward"][rew_index] += self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]

            # Reward based on positioning to spread the field.
            player_x = o['left_team'][o['active']][0]
            if player_x > 0.5: # Assuming a scale with 1 being end of field.
                components["positioning_reward"][rew_index] += self.position_reward * (player_x - 0.5)
                reward[rew_index] += components["positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
