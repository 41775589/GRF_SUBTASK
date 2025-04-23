import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a crossing and high pass focus reward."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_threshold = 0.3  # Cross-pass distance threshold
        self.high_pass_weights = 0.1  # High pass reward weight
        self.cross_threshold = 0.7  # Cross field threshold
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Add any necessary state information here
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore any necessary state here
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Adjust rewards based on crossing and high passing."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "crossing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            if obs['ball_owned_team'] == 1:  # If right team owns the ball
                last_ball_pos = obs['ball']  # get the ball's position
                if last_ball_pos[2] > 0.15 and abs(obs['ball_direction'][1]) > abs(obs['ball_direction'][0]):
                    components['high_pass_reward'][index] += self.high_pass_weights
                distance = np.linalg.norm(obs['ball'] - last_ball_pos)
                if distance > self.pass_threshold and abs(obs['ball'][0]) > self.cross_threshold:
                    components['crossing_reward'][index] += reward[index] * 0.5

            for key, value in components.items():
                reward[index] += value[index]

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
