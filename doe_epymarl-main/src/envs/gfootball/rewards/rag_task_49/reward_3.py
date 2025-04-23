import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for shooting accuracy and power from central field positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {}
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "shooting_accuracy": [0.0] * len(reward), 
                      "shooting_power": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            # Rewarding shooting accuracy
            ball_x, ball_y = obs['ball'][0], obs['ball'][1]
            distance_to_goal_y = abs(ball_y)  # Smaller y distance means better accuracy

            if obs['ball_owned_team'] == 1:  # Right team
                goal_x = 1  # Right goal x-coordinate
            else:
                goal_x = -1  # Left goal x-coordinate
            
            distance_to_goal_x = abs(ball_x - goal_x)  # Smaller x distance means closer to goal
            central_field_limit = 0.3  # Limit to define the central field region
            if -central_field_limit < ball_y < central_field_limit:
                # Increasing reward for accuracy if in central field
                components["shooting_accuracy"][rew_index] += (1 - distance_to_goal_y) * 0.1
                
                # Rewarding power based on ball speed
                ball_speed = np.linalg.norm(obs['ball_direction'])
                components["shooting_power"][rew_index] += ball_speed * 0.05

            # Combining all rewards
            reward[rew_index] += components["shooting_accuracy"][rew_index] + components["shooting_power"][rew_index]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
