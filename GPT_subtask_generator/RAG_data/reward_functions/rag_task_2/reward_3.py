import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defense-oriented checkpoint reward."""
    
    def __init__(self, env):
        super().__init__(env)
        self._defensive_checkpoints_collected = {}
        self._num_defensive_checkpoints = 5
        self._checkpoint_defensive_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_checkpoints_collected = {}
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._defensive_checkpoints_collected
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self._defensive_checkpoints_collected = from_picle['CheckpointRewardWrapper']
        return from_picle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_rewards = reward.copy()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_checkpoint_reward": [0.0] * len(reward),
        }

        if observation is None:
            return new_rewards, components

        for index in range(len(reward)):
            obs = observation[index]
            current_pos = obs['left_team'][obs['active']]
            d_to_goal = (current_pos[0] + 1) ** 2 + current_pos[1] ** 2
            num_collected_checkpoints = self._defensive_checkpoints_collected.get(index, 0)
            
            # Reward for maintaining a strategic defensive position
            distance_threshold = (0.8 / self._num_defensive_checkpoints) * num_collected_checkpoints
            if d_to_goal > distance_threshold:
                collected_now = 1
            else:
                collected_now = 0
            
            additional_reward = collected_now * self._checkpoint_defensive_reward
            components["defensive_checkpoint_reward"][index] = additional_reward
            new_rewards[index] += additional_reward
            # Update collected checkpoints
            self._defensive_checkpoints_collected[index] = min(num_collected_checkpoints + collected_now, self._num_defensive_checkpoints)

        return new_rewards, components

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
