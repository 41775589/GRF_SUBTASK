import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards focusing on shooting angles and timing."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pressure_threshold = 3  # The number of opponent players near the controlled player considered high pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_angle_reward": [0.0] * len(reward),
                      "pressure_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        # Assuming length of observation and reward match the number of agents
        for rew_index, o in enumerate(observation):
            if reward[rew_index] == 1:  # Goal scored
                components["shooting_angle_reward"][rew_index] = 1.0  # Maximize reward if scoring
                components["pressure_reward"][rew_index] = 1.0  # Additional reward for scoring under pressure
            else:
                # Analyze opponent distance to compute pressure
                opponent_distances = np.linalg.norm(o["right_team"] - o["left_team"][o["active"]], axis=1)
                pressure_index = np.sum(opponent_distances < 0.1)  # number of opponents within dangerous distance

                # Shooting angle - better angles less than 30 degrees from the central axis to the goal
                ball_position = o["ball"][:-1]  # Ignore z-axis for angle calculation
                goal_position = [1, 0] if o['ball_owned_team'] == 0 else [-1, 0] # Assuming own goal at x = -1 and opponent at x = 1
                shooting_vector = np.array(goal_position) - np.array(ball_position)
                angle = np.arccos(np.dot(shooting_vector, [1, 0]) / (np.linalg.norm(shooting_vector))) # angle with horizontal
                angle_degree = np.degrees(angle)

                # Reward for shooting angle within 30 degrees from central line to the goal
                if angle_degree < 30:
                    components["shooting_angle_reward"][rew_index] = 0.5

                # Reward when playing under pressure
                if pressure_index >= self.pressure_threshold:
                    components["pressure_reward"][rew_index] = 0.5

            # Accumulate rewards
            reward[rew_index] = reward[rew_index] + components["shooting_angle_reward"][rew_index] + components["pressure_reward"][rew_index]
                
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
