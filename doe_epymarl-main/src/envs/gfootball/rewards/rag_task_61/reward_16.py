import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a targeted reward for enhancing team synergy during possession changes, emphasizing precise timing and strategic positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.change_possession_reward = 0.2
        self.positioning_reward = 0.1
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def reward(self, reward):
        """
        Custom reward function to incentivize possession changes and strategic positioning.
        Args:
            reward (list(float)): Original reward from environment.

        Returns:
            tuple[list[float], dict[str, list[float]]]: Modified reward and reward components.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "change_possession_reward": [0.0] * len(reward), "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        agent_ids = [0, 1]  # Assumed two agents for simplicity
        current_ball_owner = observation[0]['ball_owned_team']

        for agent_id in agent_ids:
            o = observation[agent_id]

            # Reward for changing possession of the ball
            if self.previous_ball_owner is not None and self.previous_ball_owner != current_ball_owner:
                components["change_possession_reward"][agent_id] += self.change_possession_reward
                reward[agent_id] += components["change_possession_reward"][agent_id]

            # Reward for strategic positioning
            distance_from_center = np.linalg.norm(o['left_team'][o['active']] - np.array([0, 0]))
            components["positioning_reward"][agent_id] += self.positioning_reward / (1 + distance_from_center)
            reward[agent_id] += components["positioning_reward"][agent_id]

        self.previous_ball_owner = current_ball_owner

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle['previous_ball_owner']
        return from_pickle

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
