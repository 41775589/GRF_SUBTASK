import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on shooting accuracy and offensive plays involving dynamic movements."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_threshold = 0.2  # distance threshold from goal to consider it a strategic position
        self.shooting_reward_coefficient = 1.5  # increased impact for good shooting
        self.movement_reward_coefficient = 0.3  # encourage effective movement towards goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': reward.copy(),
            'shooting_reward': [0.0] * len(reward),
            'movement_reward': [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Calculate the distance to the opponent's goal (assuming right goal)
            distance_to_goal = np.abs(o['ball'][0] - 1)
            
            # Reward for shooting accuracy
            if o['ball_owned_player'] == o['active'] and distance_to_goal < self.goal_threshold:
                components['shooting_reward'][idx] = self.shooting_reward_coefficient * distance_to_goal
            
            # Evaluate movement towards the goal when possessing the ball
            if o['ball_owned_team'] == 1:  # assuming active team ID is 1
                ball_moving_towards_goal = o['ball_direction'][0] > 0  # positive x direction to right goal
                player_moving_towards_goal = o['right_team_direction'][o['active']][0] > 0
                if ball_moving_towards_goal and player_moving_towards_goal:
                    components['movement_reward'][idx] = self.movement_reward_coefficient * (1 - distance_to_goal)
            
            # Add calculated components to the reward
            reward[idx] += components['shooting_reward'][idx] + components['movement_reward'][idx]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
