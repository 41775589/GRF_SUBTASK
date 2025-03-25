import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive teamwork and strategic coordination."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_positions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_positions': self.previous_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_positions = from_pickle['CheckpointRewardWrapper']['previous_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_positioning_reward": [0.0] * len(reward)}
        
        if observation is None or len(observation) != 3:
            return reward, components

        for rew_index in range(len(reward)):
            current_obs = observation[rew_index]
            last_position = self.previous_positions.get(rew_index)
            current_position = current_obs['left_team'][current_obs['active']]
            if last_position is None:
                distance_moved = 0
            else:
                distance_moved = np.linalg.norm(last_position - current_position)
                
            # Encourage strategic defensive movements: bonus for small repositionings, penalize too much movement
            if distance_moved < 0.01:
                components['defensive_positioning_reward'][rew_index] = 0.05
            elif distance_moved > 0.1:
                components['defensive_positioning_reward'][rew_index] = -0.05
    
            # Calculate final reward
            reward[rew_index] = reward[rew_index] + components['defensive_positioning_reward'][rew_index]

            # Update previous positions
            self.previous_positions[rew_index] = current_position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
