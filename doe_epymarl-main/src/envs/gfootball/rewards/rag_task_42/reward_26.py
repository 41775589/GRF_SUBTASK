import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that applies midfield mastery and transition strategy rewards."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_control_rewards = {}
        self.number_of_midfield_zones = 5
        self.midfield_reward_value = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.midfield_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle=self.env.set_state(state)
        self.midfield_control_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'midfield_control_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            obs = observation[idx]
            player_pos = obs['left_team'][obs['active']] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']]
            
            # Calculate midfield control based on player's x position
            midfield_zone = int((player_pos[0] + 1) // (2 / self.number_of_midfield_zones))
            if obs['ball_owned_team'] == 1:  # Own the ball on midfield zone
                if midfield_zone not in self.midfield_control_rewards:
                    self.midfield_control_rewards[midfield_zone] = True
                    components['midfield_control_reward'][idx] += self.midfield_reward_value
                    reward[idx] += components['midfield_control_reward'][idx]  # add midfield transition reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        # include individual component sums in info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
