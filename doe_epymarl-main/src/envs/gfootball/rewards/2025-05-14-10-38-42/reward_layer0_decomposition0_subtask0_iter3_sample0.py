import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on offensive play coordination, particularly shooting accuracy and dynamic movements."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_reward_coefficient = 1.2  # Higher impact for accurate shooting
        self.movement_reward_coefficient = 0.5  # Encourage dynamic movements towards the goal
        self.positive_pressure_coefficient = 0.3  # Reward for managing pressure scenarios
        self.defensive_players_nearby_threshold = 0.1  # distance threshold to consider defensive pressure

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the rewards based on shooting, dynamic movements, and handling pressure."""
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': reward.copy(),
            'shooting_reward': [0.0] * len(reward),
            'movement_reward': [0.0] * len(reward),
            'pressure_handling_reward': [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for idx, o in enumerate(observation):
            goal_x = 1 if o['ball'][0] > 0 else -1  # Determine whether attacking right or left goal
            distance_to_goal = np.abs(o['ball'][0] - goal_x)
            
            # Shooting reward: Higher when close to the goal and owning the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active'] and distance_to_goal < 0.2:
                components['shooting_reward'][idx] = self.shooting_reward_coefficient * (0.2 - distance_to_goal)
                reward[idx] += components['shooting_reward'][idx]

            # Movement reward: Added when moving dynamically towards the goal with the ball possession
            moving_towards_goal = o['ball_direction'][0] * np.sign(goal_x) > 0
            if moving_towards_goal and o['ball_owned_team'] == 1:
                components['movement_reward'][idx] = self.movement_reward_coefficient * (1 - distance_to_goal)
                reward[idx] += components['movement_reward'][idx]

            # Pressure handling reward: Positive impacts when maintaining control under defensive pressure
            num_defensive_players_nearby = sum(
                np.linalg.norm(o['right_team'][i] - o['ball'][:2]) < self.defensive_players_nearby_threshold
                for i in range(len(o['right_team']))
            )
            if num_defensive_players_nearby > 0 and o['ball_owned_team'] == 1:
                components['pressure_handling_reward'][idx] = self.positive_pressure_coefficient * num_defensive_players_nearby
                reward[idx] += components['pressure_handling_reward'][idx]

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
