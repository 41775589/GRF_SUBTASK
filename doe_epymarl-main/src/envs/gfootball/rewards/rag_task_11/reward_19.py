import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that incentivizes offensive plays, fast-paced maneuvers, and precision finishing skills.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters
        self.shooting_distance_threshold = 0.3  # Ball closer than this value to the goal rewards
        self.possession_reward = 0.1            # Reward for maintaining ball possession
        self.fast_break_bonus = 0.5             # Bonus for fast approaches to the goal

    def reset(self):
        """
        Reset the environment and clear sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Augment the reward based on offensive game strategies.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        new_rewards = []

        for rew_index, r in enumerate(reward):
            o = observation[rew_index]
            goal_distance = abs(1 - o['ball'][0])  # Distance from the right goal
            ball_possession = (o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active'])

            if ball_possession:
                reward[rew_index] += self.possession_reward
            
            if goal_distance < self.shooting_distance_threshold:
                reward[rew_index] += (self.shooting_distance_threshold - goal_distance)  # Closer to goal, higher reward
            
            if ball_possession and goal_distance < 0.5 and observation[rew_index].get('ball_direction', 0)[0] > 0:
                # If possessing the ball and moving towards the goal quickly, award a fast break bonus
                reward[rew_index] += self.fast_break_bonus

            components.setdefault("goal_distance_bonus", []).append((self.shooting_distance_threshold - goal_distance) if goal_distance < self.shooting_distance_threshold else 0)
            components.setdefault("possession_bonus", []).append(self.possession_reward if ball_possession else 0)
            components.setdefault("fast_break_bonus", []).append(self.fast_break_bonus if ball_possession and goal_distance < 0.5 and observation[rew_index].get('ball_direction', 0)[0] > 0 else 0)

            new_rewards.append(reward[rew_index])

        return new_rewards, components

    def step(self, action):
        """
        Take a step using the given action and apply the custom reward transformation.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
