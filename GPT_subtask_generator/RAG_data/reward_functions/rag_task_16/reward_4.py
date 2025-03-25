import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on enhancing high passing skills in football."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_threshold = 0.3  # Assume a threshold for considering a high pass
        self.high_pass_reward = 3.0  # High reward for successful high passes
        self.target_zones = [0.75, 1.0]  # Target zones near the opponent's goal area
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
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward)
        }

        if observation is None or 'ball_direction' not in observation:
            return reward, components
        
        right_team = observation['right_team']
        ball_position = observation['ball']
        ball_direction = observation['ball_direction']
        # Check if the ball movement is upward and has significant y-axis change
        is_high_pass = (ball_direction[2] > self.pass_threshold and abs(ball_direction[1]) > self.pass_threshold)
        
        for rew_index in range(len(reward)):
            # Reward when the ball lands in target zone with a high pass
            if is_high_pass and self.target_zones[0] <= ball_position[0] <= self.target_zones[1]:
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        updated_obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in updated_obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
