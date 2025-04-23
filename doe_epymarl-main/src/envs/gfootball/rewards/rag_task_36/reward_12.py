import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for dribbling and dynamic positioning skills for football agents."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.array([0.0, 0.0])
        # Set up thresholds that define good dribbling skills and positioning
        self.dribble_reward_threshold = 0.1
        self.position_change_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.array([0.0, 0.0])
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": 0.0,
            "position_change_reward": 0.0
        }

        for o in observation:
            # Reward for dribbling: Check increased use of dribble action
            dribble_action_count = o['sticky_actions'][9]  # index 9 corresponds to dribble action
            if dribble_action_count > self.sticky_actions_counter[9]:
                components["dribble_reward"] = self.dribble_reward_threshold

            # Reward for dynamic positioning: Check for significant movement
            current_position = np.array(o['ball'])
            position_change = np.linalg.norm(current_position - self.previous_ball_position)
            if position_change > self.position_change_reward:
                components["position_change_reward"] = self.position_change_reward

            self.previous_ball_position = current_position

        total_reward = sum(components.values())
        return total_reward, components

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.previous_ball_position = from_pickle['previous_ball_position']
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
