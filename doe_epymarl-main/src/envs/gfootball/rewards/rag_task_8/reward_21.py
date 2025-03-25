import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for quick ball handling and counter-attacks after recovering possession."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recovered_possession = False
        self.ball_recovery_positions = np.zeros((2, 2))  # To store the ball position when each team recovers it
        self.recovery_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recovered_possession = False
        self.ball_recovery_positions.fill(0)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Getting the raw observation from the environment
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "recovery_reward": [0.0, 0.0]}

        # Analyze the observations for both agents
        for agent_id, obs in enumerate(observation):
            if obs['ball_owned_team'] == agent_id:
                if not self.recovered_possession:
                    # Mark as recovered possession and note the position
                    self.recovered_possession = True
                    self.ball_recovery_positions[agent_id] = obs['ball']
                else:
                    # Calculate the movement of the ball from the recovery position for the scoring
                    ball_pos_diff = np.linalg.norm(obs['ball'] - self.ball_recovery_positions[agent_id])
                    components["recovery_reward"][agent_id] = self.recovery_reward * ball_pos_diff
                    reward[agent_id] += components["recovery_reward"][agent_id]
            else:
                # When the ball is lost reset the recovery flag
                self.recovered_possession = False

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
