import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance learning of close-range attacks and decisions in football simulation."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters controlling the proximity and action based rewards
        self.goal_proximity_reward = 0.1
        self.successful_shot_reward = 1.0
        self.dribble_effectiveness_reward = 0.05
        self.opponent_goal_position = [1, 0]  # Assuming opponent goal is always at (1, 0)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_proximity_reward": [0.0] * len(reward),
                      "successful_shot_reward": [0.0] * len(reward),
                      "dribble_effectiveness_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for agent_idx, agent_obs in enumerate(observation):
            if 'ball_owned_team' in agent_obs and agent_obs['ball_owned_team'] == 1:
                # Check proximity to the opponent's goal
                distance_to_goal = np.linalg.norm(
                    np.array(self.opponent_goal_position) - np.array(agent_obs['ball']))
                proximity_factor = max(0.0, 1 - distance_to_goal)  # Normalize between 0 and 1
                components["goal_proximity_reward"][agent_idx] += \
                    self.goal_proximity_reward * proximity_factor    # Reward for being close to goal

                # Check if shooting action leads to a goal
                if agent_obs['game_mode'] == 6:  # Assuming '6' is the shot action mode
                    components["successful_shot_reward"][agent_idx] += self.successful_shot_reward

                # Check for effective dribbling action
                if agent_obs['sticky_actions'][9] == 1:  # Assuming '9' is the dribble action
                    components["dribble_effectiveness_reward"][agent_idx] += \
                      self.dribble_effectiveness_reward

            # Compute the final reward for this agent
            final_reward_component = components["base_score_reward"][agent_idx] +\
                                     components["goal_proximity_reward"][agent_idx] +\
                                     components["successful_shot_reward"][agent_idx] +\
                                     components["dribble_effectiveness_reward"][agent_idx]
            reward[agent_idx] = final_reward_component

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
