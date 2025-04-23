import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the training focusing on shot precision in tight spaces near the goal."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_precision_checkpoints = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_precision_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shot_precision_checkpoints'] = self.shot_precision_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shot_precision_checkpoints = from_pickle['shot_precision_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            components["precision_shooting_reward"][rew_index] = 0
            if 'ball_owned_team' not in o or o['ball_owned_team'] != 0:
                continue

            # Player is close to the opponent's goal and in possession of the ball
            ball_x_position = o['ball'][0]
            goal_distance_threshold = 0.2
            goal_x_position = 1  # X position of opponent's goal
            is_close_to_goal = abs(goal_x_position - ball_x_position) < goal_distance_threshold

            if is_close_to_goal:
                if rew_index not in self.shot_precision_checkpoints:
                    self.shot_precision_checkpoints[rew_index] = 1
                    shoot_actions = [football_action_set.action_shot, football_action_set.action_high_shot]
                    if "action" in o and o['action'] in shoot_actions:
                        # Reward players for attempting shots when close to goal 
                        components["precision_shooting_reward"][rew_index] = 0.5
                    else:
                        # Penalize delay or incorrect action in strategic positions
                        components["precision_shooting_reward"][rew_index] = -0.1

            reward[rew_index] += components["precision_shooting_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
