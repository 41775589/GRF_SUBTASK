import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive adaptation by rewarding precise stopping and starting movements."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.episode_steps = 0
        self.position_change_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.episode_steps = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter,
                                                'episode_steps': self.episode_steps}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.episode_steps = from_pickle['CheckpointRewardWrapper']['episode_steps']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "position_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            current_obs = observation[i]
            last_obs = self.env.unwrapped.last_observation[i] if self.env.unwrapped.last_observation else None
            
            if last_obs:
                # Calculate the Euclidean distance moved by the player
                current_pos = current_obs['left_team'][current_obs['active']] if current_obs['ball_owned_team'] == 0 else current_obs['right_team'][current_obs['active']]
                last_pos = last_obs['left_team'][last_obs['active']] if last_obs['ball_owned_team'] == 0 else last_obs['right_team'][last_obs['active']]
                
                position_change = np.linalg.norm(current_pos - last_pos)

                # Reward defensive players for significant position changes that might indicate effective press/recovery in defense
                if position_change > 0.01:  # Threshold for non-trivial movement
                    components['position_change_reward'][i] = self.position_change_reward
                    reward[i] += components['position_change_reward'][i]
                    
        # Update the counter for sticky actions
        self.episode_steps += 1
        for obs in observation:
            self.sticky_actions_counter += obs['sticky_actions']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        # Reset sticky actions counter for the next observation
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
