import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effectively using Stop-Dribble under pressure."""

    def __init__(self, env):
        super().__init__(env)
        # Counter to monitor dribble actions followed by no movement
        self.stop_dribble_counter = np.zeros(2, dtype=int)
        self.dribble_to_stop_reward = 0.2  # Reward for successful dribble to stop
        
    def reset(self):
        """Resets the environment and the stop dribble counter."""
        self.stop_dribble_counter = np.zeros(2, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Modifies the reward based on dribble to stop action completion."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {'base_score_reward': reward.copy(),
                      'dribble_stop_reward': [0.0, 0.0]}

        for idx in range(len(reward)):
            player_obs = observation[idx]

            # Check if player is active and there is change from dribbling to no action
            if player_obs['active'] != -1:
                actions = player_obs['sticky_actions']
                dribbling = actions[9]  # index 9 corresponds to dribbling action
                movement_actions = actions[0:8]
                is_moving = np.any(movement_actions)

                if dribbling and not is_moving:
                    self.stop_dribble_counter[idx] += 1

                if self.stop_dribble_counter[idx] > 1:  # i.e., dribbled then stopped
                    components['dribble_stop_reward'][idx] += self.dribble_to_stop_reward
                    reward[idx] += components['dribble_stop_reward'][idx]
                    self.stop_dribble_counter[idx] = 0  # reset the counter after reward

        return reward, components

    def step(self, action):
        """Applies the action, calculates reward, and records additional information."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return obs, reward, done, info
