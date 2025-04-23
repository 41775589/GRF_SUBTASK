import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for adding rewards based on effective ball clearance from defensive zones under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_count = np.zeros(2, dtype=int)  # Tracking clearance per team side - 0: left to right, 1: right to left
        self.clearance_threshold = 0.2  # Minimum distance to recognize as a clearance
        self.pressure_threshold = 0.5  # Distance defining 'under pressure'
        self.reward_for_clearance = 0.2

    def reset(self):
        """Reset the environment and the clearance counts."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_count.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state including clearance data."""
        to_pickle['clearance_counts'] = self.clearance_count.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state including clearance data."""
        from_pickle = self.env.set_state(state)
        self.clearance_count = from_pickle['clearance_counts']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on effective clearance under pressure."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, r in enumerate(reward):
            o = observation[rew_index]
            if not o:
                continue
            
            # Calculate distances
            ball_position = o['ball'][:2]
            player_positions = o['left_team' if (rew_index == 0) else 'right_team']
            closest_player_distance = np.min(np.linalg.norm(player_positions - ball_position, axis=1))

            last_ball_position = self.prev_ball_position if hasattr(self, 'prev_ball_position') else ball_position
            ball_moved_towards_goal = np.linalg.norm([1, 0] - ball_position) < np.linalg.norm([1, 0] - last_ball_position)

            if closest_player_distance < self.pressure_threshold:
                if ball_moved_towards_goal and np.linalg.norm(ball_position) > self.clearance_threshold:
                    components["clearance_reward"][rew_index] = self.reward_for_clearance
                    reward[rew_index] += components["clearance_reward"][rew_index]
                    self.clearance_count[rew_index] += 1

        self.prev_ball_position = ball_position

        return reward, components

    def step(self, action):
        """Wrap environment's step function to add custom reward information."""
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
