import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dense rewards based on offensive strategies 
    including effective shooting, dribbling and passing."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "shoot_efficiency": [0.0] * len(reward),
                      "dribble_efficiency": [0.0] * len(reward), "pass_efficiency": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            base_reward = reward[rew_index]
            components["shoot_efficiency"][rew_index] = self.calculate_shooting_efficiency(obs)
            components["dribble_efficiency"][rew_index] = self.calculate_dribble_efficiency(obs)
            components["pass_efficiency"][rew_index] = self.calculate_pass_efficiency(obs)

            reward[rew_index] += (components["shoot_efficiency"][rew_index] +
                                  components["dribble_efficiency"][rew_index] +
                                  components["pass_efficiency"][rew_index])

        return reward, components
    
    def calculate_shooting_efficiency(self, obs):
        if obs['game_mode'] == 4:  # Assuming mode 4 is a shooting scenario
            if obs['ball_owned_team'] == 0:  # Agents' team owns the ball
                # Simplified efficiency calculation: closer to goal, higher reward
                x_pos = obs['ball'][0]
                return (1 - abs(1 - x_pos)) * 0.1  # Scale factor
        return 0.0

    def calculate_dribble_efficiency(self, obs):
        # Assuming a higher dribble (action 9) count with ball possession indicates effective dribbling
        if obs['ball_owned_team'] == 0:  # Agents' team owns the ball
            dribble_count = obs['sticky_actions'][9]  # Dribbling action index
            dribble_efficiency = 0.05 * dribble_count
            return min(dribble_efficiency, 0.1)  # Cap dribbling reward
        return 0.0

    def calculate_pass_efficiency(self, obs):
        # Simplified: reward for executing long/high passes (hypothetical sticky_actions indices 7 and 8)
        if obs['ball_owned_team'] == 0:  # Agents' team owns the ball
            long_pass_count = obs['sticky_actions'][7]
            high_pass_count = obs['sticky_actions'][8]
            return 0.05 * (long_pass_count + high_pass_count)  # Weight for each completed pass
        return 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Apply reward transformation
        reward, reward_components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter for more nuanced training data
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_state
        return obs, reward, done, info
