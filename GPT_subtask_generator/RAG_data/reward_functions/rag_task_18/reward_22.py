import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward based on controlled transitions and pace management."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_rewards = np.zeros((3, 3), dtype=float)  # Rewards for controlled transitions between players
        self.pace_rewards = np.zeros(3, dtype=float)  # Rewards for maintaining optimal pacing

        # Initialize the control transition matrix slightly favoring MID field transitions
        self.initialize_transition_rewards()
        # Initialize pace control rewards
        self.initialize_pace_rewards()

    def initialize_transition_rewards(self):
        """Higher rewards for transitions between midfield players."""
        self.transition_rewards.fill(0.01)  # baseline transition reward
        # Enhance incentives for midfield connections
        mid_roles = [4, 5, 6, 8]  # Considering CM, DM, LM, AM as central roles
        for i in mid_roles:
            for j in mid_roles:
                self.transition_rewards[i, j] = 0.05

    def initialize_pace_rewards(self):
        """Reward designed to promote effective pace control."""
        optimal_pace = 0.1  # Example pace considered optimal
        self.pace_rewards.fill(optimal_pace)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Calculate modified reward with transition and pace adjustments."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": np.zeros(len(reward)),
                      "pace_reward": np.zeros(len(reward))}

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            # Handling ball transitions
            if player_obs['ball_owned_team'] == 0:  # Assuming '0' is the team of the agent
                pass_player = player_obs['ball_owned_player']
                active_player = player_obs['active']
                components["transition_reward"][rew_index] = self.transition_rewards[pass_player, active_player]

            # Handling pace control
            current_pace = np.linalg.norm(player_obs['ball_direction'][:2])
            components["pace_reward"][rew_index] = -abs(current_pace - self.pace_rewards[active_player])

            # Total reward modification
            reward[rew_index] += (components["transition_reward"][rew_index] +
                                  components["pace_reward"][rew_index])

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
