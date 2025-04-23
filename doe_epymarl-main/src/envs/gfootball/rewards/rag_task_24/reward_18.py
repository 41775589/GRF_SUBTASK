import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances rewards based on effective long-range passing through strategically important areas of the field.
    """
    def __init__(self, env):
        super().__init__(env)
        self.pass_checkpoints = [
            [-0.4, 0],      # Midfield left
            [0, 0],         # Midfield center
            [0.4, 0],       # Midfield right
            [-0.75, 0.25],  # Left Forward
            [0.75, 0.25],   # Right Forward
            [-0.75, -0.25], # Left Backward
            [0.75, -0.25]   # Right Backward
        ]
        self.pass_threshold = 0.1
        self.pass_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for agent_idx, agent_reward in enumerate(reward):
            o = observation[agent_idx]
            ball_pos = o['ball'][:2]
            last_ball_pos = o['ball_direction'][:2] + ball_pos  # Approximate the previous ball position

            # Identify valid ball passes landing near checkpoints
            for cp in self.pass_checkpoints:
                if np.linalg.norm(np.array(cp) - last_ball_pos) <= self.pass_threshold:
                    if np.linalg.norm(np.array(cp) - ball_pos) <= self.pass_threshold:
                        # Reward only passes that travel sufficient distance for strategic positioning
                        if np.linalg.norm(ball_pos - last_ball_pos) > self.pass_threshold * 5:
                            components['pass_reward'][agent_idx] += self.pass_reward
                            reward[agent_idx] += self.pass_reward

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
