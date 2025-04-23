import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward to promote fluid transitions between defense and offense through dribbling maneuvers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.position_rewards_collected = set()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_reward = 0.05
        self.position_change_reward = 0.1

    def reset(self):
        self.position_rewards_collected = set()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = list(self.position_rewards_collected)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_rewards_collected = set(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'dribble_reward': [0.0] * len(reward),
                      'position_change_reward': [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            current_position = (o['active'], tuple(o['left_team'][o['active']]) if o['ball_owned_team'] == 0 else tuple(o['right_team'][o['active']]))

            # Reward for dribbling control
            if o['sticky_actions'][9] == 1:  # action_dribble is the 10th action (index 9)
                components['dribble_reward'][rew_index] = self.dribble_reward
                reward[rew_index] += components['dribble_reward'][rew_index]
            
            # Reward for changing positions after dribbling
            if current_position not in self.position_rewards_collected:
                if self.sticky_actions_counter[9] > 0:  # Check if dribble action was previously active
                    components['position_change_reward'][rew_index] = self.position_change_reward
                    reward[rew_index] += components['position_change_reward'][rew_index]
                    self.position_rewards_collected.add(current_position)
        
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
            for i, active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = active
                self.sticky_actions_counter[i] += active
        return observation, reward, done, info
