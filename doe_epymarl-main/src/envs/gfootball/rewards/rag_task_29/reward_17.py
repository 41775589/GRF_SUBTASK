import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on enhancing shot precision in close-range scenarios."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define distances and angles for tight space shooting scenarios
        self.goal_distance_threshold = 0.2  # Close proximity to the goal
        self.angle_reward_scale = 1.0       # Scale for rewarding good angles
        self.power_reward_scale = 1.0       # Scale for rewarding optimal power

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_action_counters'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle.get('sticky_action_counters', []), dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            player_obs = observation[i]
            ball_pos = player_obs['ball']

            # Check if the player is in the opposing team's goal area
            if abs(ball_pos[0]) > (1 - self.goal_distance_threshold):
                components["precision_shooting_reward"][i] = self.angle_reward_scale * self.evaluate_angle(player_obs)
                components["precision_shooting_reward"][i] += self.power_reward_scale * self.evaluate_power(player_obs)
                reward[i] += components["precision_shooting_reward"][i]

        return reward, components

    def evaluate_angle(self, player_obs):
        """Evaluates the player's shooting angle; closer to perpendicular is better."""
        ball_dir = player_obs['ball_direction']
        angle_to_goal = np.arctan2(ball_dir[1], ball_dir[0])
        angle_reward = np.cos(angle_to_goal)  # Cosine of angle to goal-axis, maximized when shooting straight
        # Normalize to [0, 1]
        return (angle_reward + 1) / 2 

    def evaluate_power(self, player_obs):
        """Evaluates the shooting power; optimal power for close-range shots."""
        # Simplified power evaluation (no actual power data available in dataset)
        ball_speed = np.linalg.norm(player_obs['ball_direction'][0:2])
        optimal_speed = 0.1  # assuming some optimal speed, to be tuned/learned
        power_penalty = np.abs(ball_speed - optimal_speed)
        return 1 - power_penalty  # rewarding closer speeds to the optimal

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Add components and other tracking info to debug outputs
        for name, values in components.items():
            info[f"component_{name}"] = sum(values)
        obs = self.env.unwrapped.observation()
        if obs is not None:
            for agent_obs in obs:
                for i, action_active in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += int(action_active)
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
