import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards for shooting accuracy and power from central field positions in a soccer game."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the x-range of central field positions
        self.central_field_range = (-0.25, 0.25)  # Central x-range
        self.power_shot_threshold = 0.7  # Threshold for considering a shot as powerful

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_central = self.central_field_range[0] <= o['ball'][0] <= self.central_field_range[1]
            is_powerful = np.linalg.norm(o['ball_direction'][:2]) >= self.power_shot_threshold
            own_goal_range = self._opponent_goal_range(o['ball_owned_team'])

            # Check if ball is shot from central field towards opponent's goal with sufficient power
            if is_central and is_powerful and o['ball'][0] in own_goal_range:
                components["position_reward"][rew_index] = 0.5  # Provide additional reward
                reward[rew_index] += 1.5 * components["position_reward"][rew_index]

        return reward, components

    def _opponent_goal_range(self, team):
        """Helper to determine the opponent's goal range based on which team owns the ball."""
        if team == 0:
            return (0.9, 1.0)  # Approximation of right side goal range
        elif team == 1:
            return (-1.0, -0.9)  # Approximation of left side goal range
        return (0, 0)  # Neutral or error state, no goal possible

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
