import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward structure to focus on varied shooting techniques from different field positions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_multiplier = 1.5  # Higher rewards for scoring from distance shots
        self.manual_control_bonus = 0.2  # Bonuses for shots controlled manually

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "distance_shot_bonus": [0.0],
            "manual_control_bonus": [0.0]
        }
        
        player_pos = observation['left_team'][observation['active']]  # Assuming single agent control
        ball_pos = observation['ball'][:2]  # X, Y coordinates of the ball
        goal_pos = [1, 0]  # Right side goal position

        # Calculate distance from player to goal
        distance = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))
        
        # Reward for long distance shots
        if distance > 0.65:
            components["distance_shot_bonus"][0] = self.reward_multiplier
            reward[0] += components["distance_shot_bonus"][0]
        
        # Bonus for manual control (when player uses dribble or sprint effectively)
        if observation['sticky_actions'][8] == 1 or observation['sticky_actions'][9] == 1:  # Check for sprint or dribble action
            components["manual_control_bonus"][0] = self.manual_control_bonus
            reward[0] += components["manual_control_bonus"][0]
        
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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
