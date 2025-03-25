import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive training by rewarding tactical responses and quick transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_defensive_positions = 8  # Defensive strategic positions for simulating counter-attacks
        self.defensive_rewards = np.zeros(self.num_defensive_positions)
        self.transition_reward = 0.1

    def reset(self):
        """Reset sticky action counters and defensive rewards for the new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include defensive rewards state in pickle."""
        to_pickle['defensive_rewards'] = self.defensive_rewards.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Extract defensive rewards state from pickle."""
        from_pickle = self.env.set_state(state)
        self.defensive_rewards = from_pickle['defensive_rewards']
        return from_pickle

    def reward(self, reward):
        """Compute rewards based on position and ball ownership transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_transition_reward": [0.0] * len(reward)}

        for idx, rew in enumerate(reward):
            components["defensive_transition_reward"][idx] = 0
            o = observation[idx]
            if o['ball_owned_team'] == 0:  # If ball is with our team (defensive focus)
                player_pos = o['left_team'][o['active']]
                opponent_goal_pos = 1.0

                # Reward defensive positioning behind the ball
                if player_pos[0] < o['ball'][0]:  # Player must be behind the ball
                    distance_to_goal = np.abs(player_pos[0] - opponent_goal_pos)
                    defensive_idx = int(distance_to_goal // 0.125)  # Discretize the field
                    if defensive_idx < self.num_defensive_positions and self.defensive_rewards[defensive_idx] == 0:
                        components["defensive_transition_reward"][idx] = self.transition_reward
                        self.defensive_rewards[defensive_idx] = 1

            # Total modified reward
            reward[idx] += components["defensive_transition_reward"][idx]

        return reward, components

    def step(self, action):
        """Performs one timestep of the environment's dynamics."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, sa in enumerate(agent_obs['sticky_actions']):
                if sa:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = sa
        return obs, reward, done, info
