import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Custom reward function for enhancing shot precision skills specifically for scenarios
    within close range of the goal, including angles and power adjustments required to beat
    the goalkeeper from a tight space."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Parameters for the reward function
        self.goal_proximity_threshold = 0.15  # Distance threshold to consider "close to goal"
        self.precision_bonus = 0.5            # Bonus reward for exact shot at the goal
        self.shot_attempt_penalty = -0.1      # Penalty when a shot attempt fails

        # To track sticky actions (like dribble or sprint decisions)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_bonus": [0.0] * len(reward),
                      "shot_attempt_penalty": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            distance_to_goal = np.linalg.norm(o['ball'][:2] - [1, 0])
            own_goal = o['ball_owned_team'] == 1 and o['active']
            
            # Check if the agent is close to opponent's goal and owns the ball
            if own_goal and distance_to_goal < self.goal_proximity_threshold:
                # Calculate angle to the center of the goalpost at y = 0
                dir_vector = o['ball_direction'][:2]
                goal_vector = np.array([1, 0]) - o['ball'][:2]
                cos_angle = np.dot(dir_vector, goal_vector) / (np.linalg.norm(dir_vector) * np.linalg.norm(goal_vector))
                
                # Reward players for precise aiming directly at the goal center
                if cos_angle > 0.9:  # Approximately corresponds to about +/- 25 degrees deviation
                    components["precision_bonus"][rew_index] = self.precision_bonus
                    reward[rew_index] += components["precision_bonus"][rew_index]
                elif cos_angle < 0.5:  # If the shot is not well directed, we penalize slightly
                    components["shot_attempt_penalty"][rew_index] = self.shot_attempt_penalty
                    reward[rew_index] += components["shot_attempt_penalty"][rew_index]

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
