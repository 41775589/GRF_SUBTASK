import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a rewards focused on defensive football skills."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_positions = np.zeros(2, dtype=(float, 2))
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for specific defensive actions
        self.interception_radius = 0.01  # Radius in which an interception is considered
        self.tackling_distance = 0.02    # Distance threshold for successful tackling

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_positions.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['InterceptorRewardWrapper'] = self.previous_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_positions = from_pickle['InterceptorRewardWrapper']
        return from_pickle

    def reward(self, reward):
        current_positions = self.env.render(mode='state_pixels')
        observation = self.env.unwrapped.observation()
        updated_rewards = reward.copy()
        components = {"base_score_reward": reward.copy(), "defensive_action_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            player_pos = np.array(o['left_team'][o['active']])
            current_ball_pos = o['ball'][:2]

            # Reward intercepting by checking if player movement corresponds to ball's last known movement
            if np.linalg.norm(player_pos - current_ball_pos) <= self.interception_radius:
                components["defensive_action_reward"][idx] += 1.0

            # Reward for tackling/sliding, if the player performed a sliding action and was near the opponent
            if o['sticky_actions'][9] == 1 and self._check_tackle_proximity(player_pos, o['right_team']):
                components["defensive_action_reward"][idx] += 1.5

            updated_rewards[idx] += components["defensive_action_reward"][idx]

        self.previous_positions = current_positions
        return updated_rewards, components

    def _check_tackle_proximity(self, player_pos, opponent_positions):
        """Check if player is close enough to opponents for a successful tackle."""
        for opp_pos in opponent_positions:
            if np.linalg.norm(player_pos - opp_pos) < self.tackling_distance:
                return True
        return False

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
