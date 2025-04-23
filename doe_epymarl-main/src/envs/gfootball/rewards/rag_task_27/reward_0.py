import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to add a dense reward based on defensive maneuvers and interceptions."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize sticky actions counter to record the frequency of specific defensive actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset the sticky actions counter at the start of each episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            components["defense_position_reward"][i] = self.calculate_defensive_reward(o)

            # Update the total reward for each agent based on defense positioning
            reward[i] += components["defense_position_reward"][i]

        return reward, components

    def calculate_defensive_reward(self, obs):
        """Calculate rewards based on defensive positioning and interceptions."""
        reward = 0
        # If the ball is close to their goal and they are intercepting
        if obs['ball_owned_team'] != 1 and np.linalg.norm(obs['ball']) < 0.3:
            reward += 0.2  # Ball closer to own goal gives extra reward

        # Additional reward if they just intercepted the ball from the opposition
        if obs['ball_owned_team'] == 0 and self.env.previous_ball_owned_team == 1:
            reward += 0.5

        return reward

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
