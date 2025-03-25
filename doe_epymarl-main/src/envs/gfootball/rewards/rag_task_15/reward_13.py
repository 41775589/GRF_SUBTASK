import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on the precision and length of passes in a football game."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_rewards = {}
        self.long_pass_distance_threshold = 0.5  # Long pass threshold in normalized field units
        self.high_precision_threshold = 0.1       # Precision threshold relative to player positions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.pass_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Calculate distance of ball travel
            ball_travel_distance = np.linalg.norm(o['ball_direction'][:2])

            if ball_travel_distance > self.long_pass_distance_threshold:
                # Check for pass precision based on ball's direction and distance to teammates
                precision = min(np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1))
                if precision < self.high_precision_threshold:
                    components["pass_reward"][rew_index] = 0.5  # Reward for successful long, precise pass
                    reward[rew_index] += components["pass_reward"][rew_index]
            
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
