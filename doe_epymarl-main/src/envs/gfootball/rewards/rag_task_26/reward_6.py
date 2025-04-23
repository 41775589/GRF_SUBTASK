import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards specific midfield dynamics and ball control."""
    
    def __init__(self, env):
        super().__init__(env)
        self.midfield_checkpoints = [0.2, 0.4, 0.6, 0.8]  # normalized midfield regions
        self.checkpoint_rewards = {cp: False for cp in self.midfield_checkpoints}
        self.checkpoint_value = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoint_rewards = {cp: False for cp in self.midfield_checkpoints}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoint_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoint_rewards = from_pickle.get('CheckpointRewardWrapper', 
                                                  {cp: False for cp in self.midfield_checkpoints})
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            x_pos = o['ball'][0]
            
            # Apply midfield checkpoint rewards based on ball position
            for cp in self.midfield_checkpoints:
                if (not self.checkpoint_rewards[cp]) and cp <= x_pos <= cp + 0.2:
                    if ('ball_owned_team' not in o or o['ball_owned_team'] != 0):  # Player's team owns the ball
                        components["checkpoint_reward"][rew_index] = self.checkpoint_value
                        reward[rew_index] += self.checkpoint_value
                        self.checkpoint_rewards[cp] = True

            # Enhance rewards for maintaining possession and moving towards forward
            if o['ball_owned_team'] == 0:  # Assuming 0 is the player's team
                if x_pos > 0.6:  # Moving towards opposition goal
                    components["checkpoint_reward"][rew_index] += self.checkpoint_value
                    reward[rew_index] += self.checkpoint_value

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        if 'sticky_actions' in observation[-1]:
            for i in range(len(observation[-1]['sticky_actions'])):
                self.sticky_actions_counter[i] = observation[-1]['sticky_actions'][i]
        return obs, reward, done, info
