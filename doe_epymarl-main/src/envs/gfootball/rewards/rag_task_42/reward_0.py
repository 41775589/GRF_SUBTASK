import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on midfield dynamics, focusing on both pressure handling 
    and strategic positioning during offensive and defensive transitions.
    """
    def __init__(self, env):
        super().__init__(env)
        # Tracking previous states for midfield dynamics
        self.previous_ball_position = None
        self.transition_zones = [
            [-0.3, 0.3], # Midfield zone in x-direction
        ]
        self.in_transition_zone = False
        self.pressure_coefficient = 1.0
        self.repositioning_coefficient = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.in_transition_zone = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self.previous_ball_position,
            'in_transition_zone': self.in_transition_zone
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle['CheckpointRewardWrapper']
        self.previous_ball_position = saved_state['previous_ball_position']
        self.in_transition_zone = saved_state['in_transition_zone']
        return from_pickle

    def reward(self, reward):
        current_observation = self.env.unwrapped.observation()
        if current_observation is None:
            return reward, {}

        ball_position = current_observation['ball'][:2] # Only consider x, y
        ball_owned_team = current_observation['ball_owned_team']
        
        components = {"base_score_reward": reward.copy(), "pressure_reward": 0.0, "repositioning_reward": 0.0}
        
        if self.previous_ball_position is None:
            self.previous_ball_position = ball_position
        
        ball_distance_moved = np.linalg.norm(np.array(ball_position) - np.array(self.previous_ball_position))
        self.previous_ball_position = ball_position

        if ball_owned_team != -1:  # If the ball is owned, we check for pressure
            # Add pressure based reward if the agent has the ball and enemy players are nearby
            own_team = 'left_team' if ball_owned_team == 0 else 'right_team'
            enemy_team = 'right_team' if ball_owned_team == 0 else 'left_team'
            own_player_positions = current_observation[own_team]
            enemy_player_positions = current_observation[enemy_team]

            # Average distance of enemy players to the ball
            average_enemy_distance = np.mean([np.linalg.norm(enemy_pos - ball_position) for enemy_pos in enemy_player_positions])
            if average_enemy_distance < 0.3:  # arbitrary threshold for "pressure"
                components['pressure_reward'] = self.pressure_coefficient * (0.3 - average_enemy_distance)
        
        # Reward for strategic repositioning in transitions
        if self.in_transition_zone:
            # If was in transition zone and now is closer to goal area or further from own goal
            if (any((zone[0] <= ball_position[0] <= zone[1]) for zone in self.transition_zones) and
               ball_distance_moved > 0.05):  # Arbitrarily define significant move
                components['repositioning_reward'] = self.repositioning_coefficient * ball_distance_moved
                self.in_transition_zone = False
        else:
            if any((zone[0] <= ball_position[0] <= zone[1]) for zone in self.transition_zones):
                self.in_transition_zone = True
        
        total_reward = sum(reward) + components['pressure_reward'] + components['repositioning_reward']

        return [total_reward], components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action

        return obs, reward, done, info
