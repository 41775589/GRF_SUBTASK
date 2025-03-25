import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on midfield wide positioning and successful high passes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_success_count = np.zeros(2, dtype=int)  # Two agents: left and right
        self.position_reward_multiplier = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_success_count.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['high_pass_success_count'] = self.high_pass_success_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.high_pass_success_count = from_pickle['high_pass_success_count']
        return from_pickle

    def reward(self, reward):
        # Update reward based on mid-field wide positioning and successful high passes
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # High Pass was successful
            if 'high_pass' in o['action'] and o['ball_owned_team'] == rew_index:
                self.high_pass_success_count[rew_index] += 1
            
            # Reward lateral positioning for midfield wide players
            mid_fielder_positions = np.where(o['left_team_roles'] == 6) + np.where(o['left_team_roles'] == 7)
            for pos_index in mid_fielder_positions:
                player_y_position = abs(o['left_team'][pos_index][1])  # Get y-position
                components["positioning_reward"][rew_index] += player_y_position * self.position_reward_multiplier

            # Calculate total reward
            components["high_pass_reward"][rew_index] = 0.1 * self.high_pass_success_count[rew_index]
            reward[rew_index] += components["positioning_reward"][rew_index] + components["high_pass_reward"][rew_index]

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
