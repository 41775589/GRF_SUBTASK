import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for successful long-range shots and good positioning for these shots."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distance_thresholds = [0.7, 0.85]  # Thresholds defining long-range distances
        self.distance_rewards = {0.7: 0.1, 0.85: 0.3}  # Rewards for reaching specific distances from the goal
        self.shot_reward = 1.0  # Reward for taking a shot from long range

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        # This function could store relevant state data as required
        return to_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # This function could restore state data as required
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Compute the Euclidean distance to the opponent's goal
            goal_x = 1.0  # Opponent's goal X coordinate
            position_x = o['right_team'][o['active']][0]  # X position of active player
            distance_to_goal = abs(goal_x - position_x)
            
            # Provide rewards for reaching specific distances from the goal
            for threshold in self.distance_thresholds:
                if distance_to_goal > threshold:
                    components["positioning_reward"][rew_index] += self.distance_rewards[threshold]
            
            # Reward for taking a shot from a long range
            if 'action' in o and o['action'] == 'shot' and distance_to_goal > self.distance_thresholds[0]:
                components["shot_reward"][rew_index] += self.shot_reward

            # Update total reward
            reward[rew_index] += components["positioning_reward"][rew_index] + components["shot_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
