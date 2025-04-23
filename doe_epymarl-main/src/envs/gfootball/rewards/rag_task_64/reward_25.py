import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on high passes and crossing from varying distances and angles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_weight = 3.0  # Reward weighting for successful high passes
        self.crossing_angle_weight = 2.0  # Reward weight for crossing angles

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": np.array(reward).copy(),
            "pass_quality_reward": np.zeros_like(reward),
            "crossing_angle_reward": np.zeros_like(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if high passes are well executed
            if o['game_mode'] == 3:
                components["pass_quality_reward"][rew_index] = self.pass_quality_weight * 1  # example weight
                reward[rew_index] += components["pass_quality_reward"][rew_index]

            # Evaluate crossing from different angles
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                ball_pos = o['ball']
                goal_pos = [1, 0]  # Assuming crossing to the right goal

                delta_x, delta_y = goal_pos[0] - ball_pos[0], goal_pos[1] - ball_pos[1]
                angle = np.arctan2(delta_y, delta_x)
                if np.abs(angle) > np.pi / 4:  # Arbitrary angle threshold
                    components["crossing_angle_reward"][rew_index] = self.crossing_angle_weight * (np.abs(angle) - np.pi/4)
                    reward[rew_index] += components["crossing_angle_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
