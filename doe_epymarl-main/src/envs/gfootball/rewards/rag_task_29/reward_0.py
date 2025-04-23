import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing shot precision from close ranges."""
    def __init__(self, env):
        super().__init__(env)
        self.checkpoint_radius = 0.1  # Tight range around the goal
        self.goal_position = np.array([1, 0])  # Goal position (right-side goal)
        self.precision_multiplier = 5  # Multiplier for precise shots
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "precision_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        # Assume length of reward corresponds to number directly-controlled agents
        for idx, single_reward in enumerate(reward):
            o = observation[idx]
            
            # Only proceed if the active player's team owns the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                ball_pos = o['ball'][:2]  # Get the x, y position of the ball
                distance_to_goal = np.linalg.norm(self.goal_position - ball_pos)

                # Check if the ball is within the close range of the goal
                if distance_to_goal <= self.checkpoint_radius:
                    angle_with_goal = np.arctan2(self.goal_position[1] - ball_pos[1],
                                                 self.goal_position[0] - ball_pos[0])
                    goal_facing_direction = np.arctan2(o['right_team_direction'][o['active'], 1],
                                                       o['right_team_direction'][o['active'], 0])

                    angle_diff = abs(angle_with_goal - goal_facing_direction)

                    # Reward based on facing the goal correctly while close
                    if angle_diff < np.radians(30):  # within +/- 30 degrees towards goal
                        precision_bonus = self.precision_multiplier / (angle_diff + 0.01)  # Scaled on precision
                        components['precision_bonus'][idx] = precision_bonus
                        reward[idx] += precision_bonus
            
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
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
