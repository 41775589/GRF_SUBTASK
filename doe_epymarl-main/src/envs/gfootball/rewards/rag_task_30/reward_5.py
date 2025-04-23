import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strategic defensive positioning and transitioning to counterattack."""

    def __init__(self, env):
        super().__init__(env)
        self.last_ball_position_x = 0
        self.transition_reward = 0.1
        self.defensive_position_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.last_ball_position_x = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 0:  # if left team has the ball
                current_ball_x = o['ball'][0]
                
                # Check if moving defensively (back or laterally, not owning the ball)
                if o['ball_owned_player'] != o['active'] and self.last_ball_position_x >= current_ball_x:
                    components['defensive_position_reward'][rew_index] = self.defensive_position_reward
                    reward[rew_index] += components['defensive_position_reward'][rew_index]
                
                # Check for transition from defense to attack (counterattack):
                if abs(self.last_ball_position_x - o['left_team'][o['active']][0]) > 0.5:
                    components['transition_reward'][rew_index] = self.transition_reward
                    reward[rew_index] += components['transition_reward'][rew_index]
                
                self.last_ball_position_x = current_ball_x  # Update the last ball position

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
