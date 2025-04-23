import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for close-range attacks, emphasizing on agility and quick decision-making."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.close_range_reward = 0.2
        self.goal_zone_threshold = 0.2  # Threshold for goal zone proximity

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickled = self.env.set_state(state)
        self.sticky_actions_counter = from_pickled.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickled

    def reward(self, reward):
        """Augment reward based on the player's proximity to the goal when taking a shot or dribbling."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "close_range_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for idx, obs in enumerate(observation):
            ball_pos = obs['ball'][0]  # Only consider x coordinate for proximity on goal direction
            is_shot = obs['sticky_actions'][9]  # Assuming index 9 is the shot action
            is_dribbling = obs['sticky_actions'][8]  # Assuming index 8 is dribbling action
            
            if abs(ball_pos) > (1 - self.goal_zone_threshold):
                if is_shot or is_dribbling:
                    extra_reward = self.close_range_reward * (1 - abs(ball_pos))
                    components["close_range_reward"][idx] = extra_reward
                    reward[idx] += extra_reward

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
