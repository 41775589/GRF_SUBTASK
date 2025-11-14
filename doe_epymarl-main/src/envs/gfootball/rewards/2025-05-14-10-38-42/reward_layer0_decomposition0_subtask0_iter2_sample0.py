import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on offensive play coordination, particularly shooting accuracy and dynamic movements."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_reward_coefficient = 1.2  # Higher impact for accurate shooting
        self.movement_reward_coefficient = 0.5  # Encourage dynamic movements towards the goal

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the rewards based on shooting and dynamic movements towards the goal."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'shooting_reward': [0.0] * len(reward),
                      'movement_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for idx, o in enumerate(observation):
            goal_x_coord = 1 if o['ball'][0] > 0 else -1  # Determine which goal to attack based on ball position
            distance_to_goal = np.abs(o['ball'][0] - goal_x_coord)
            
            # Shooting reward: Increased when close to the goal and possession of the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active'] and distance_to_goal < 0.2:
                components['shooting_reward'][idx] = self.shooting_reward_coefficient * (0.2 - distance_to_goal)
                reward[idx] += components['shooting_reward'][idx]
            
            # Movement reward: Added when moving towards the goal with the ball
            moving_towards_goal = o['ball_direction'][0] * np.sign(goal_x_coord) > 0
            if moving_towards_goal and o['ball_owned_team'] == 1:
                components['movement_reward'][idx] = self.movement_reward_coefficient * (1 - distance_to_goal)
                reward[idx] += components['movement_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f'sticky_actions_{i}'] = self.sticky_actions_counter[i]
        return observation, reward, done, info
