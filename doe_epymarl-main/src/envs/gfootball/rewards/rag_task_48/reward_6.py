import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing successful high passes from midfield, enhancing scoring opportunities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_threshold = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # We don't need to restore any persistent state for this task-specific reward wrapper.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            # Check if the ball is in midfield and if a high pass has been initiated
            if abs(o['ball'][0]) < self.midfield_threshold and o['ball_owned_team'] == 0:
                ball_velocity_z = o['ball_direction'][2]
                if ball_velocity_z > 0.1:  # Assuming upward velocity indicates a high pass.
                    # Reward depends on the proximity of receiving player to the opponent's goal area
                    opponent_goal_x = 1
                    closest_player_dist = min(np.linalg.norm(o['right_team'][:, :2] - [opponent_goal_x, 0], axis=1))
                    score_opportunity_factor = max(0, 1 - closest_player_dist)
                    components["high_pass_reward"][rew_index] = 0.5 * score_opportunity_factor  # coefficient 0.5 for tuning

            reward[rew_index] += components["high_pass_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
