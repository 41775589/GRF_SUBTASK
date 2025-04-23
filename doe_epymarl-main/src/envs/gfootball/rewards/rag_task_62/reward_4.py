import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting optimization reward around the goal area."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Divide the region near the goal into sectors for angle optimization
        self._goal_sectors = 5
        self._sector_reward = 0.2
        self._pressure_reward = 0.5
        self._min_distance = 0.2  # minimum distance to goal to consider
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for checkpoint data, if any."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state for checkpoint data, if any."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on shooting optimization criteria."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['game_mode'] != 0:  # Only consider normal gameplay
                continue

            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                ball_pos = o['ball'][:2]
                goal_pos = [1, 0]  # Right team's goal position on the x-axis
                dist_to_goal = np.linalg.norm(np.array(ball_pos) - np.array(goal_pos))
                
                if dist_to_goal < self._min_distance:
                    angle = np.arctan2(ball_pos[1], ball_pos[0] - goal_pos[0])
                    sector_index = (int((angle + np.pi) / (2 * np.pi / self._goal_sectors))) % self._goal_sectors
                    reward[rew_index] += self._sector_reward

                    # Check for defensive pressure
                    for defender_pos in o['left_team']:
                        if np.linalg.norm(np.array(ball_pos) - np.array(defender_pos)) < 0.1:  # within pressure distance
                            reward[rew_index] += self._pressure_reward
                            break

        return reward, components

    def step(self, action):
        """Take a step in the environment, modifying reward based on custom logic."""
        observation, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        info["final_reward"] = sum(modified_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, modified_reward, done, info
