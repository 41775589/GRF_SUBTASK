import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adapts and enhances rewards based on specialized tactics in football."""

    def __init__(self, env):
        super().__init__(env)
        # Adjusting reward coefficients based on previous analysis
        self.pass_reward_coef = 0.2  # Reduced to prevent overpowering other components
        self.shot_reward_coef = 3.0  # Increased to encourage shooting due to previous 0s
        self.dribble_reward_coef = 0.2  # Adjusted for better balance
        self.position_reward_coef = 0.5  # New reward component for good positioning

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": np.zeros(len(reward)),
            "shot_reward": np.zeros(len(reward)),
            "dribble_reward": np.zeros(len(reward)),
            "position_reward": np.zeros(len(reward))
        }

        for i, o in enumerate(obs):
            if o['game_mode'] == 0 and o['ball_owned_team'] == 1:  # In-play and ball owned by our team
                if o['sticky_actions'][1] == 1:  # Short pass
                    components["pass_reward"][i] = self.pass_reward_coef
                if o['sticky_actions'][8] == 1:  # Long pass
                    components["pass_reward"][i] += self.pass_reward_coef

                if o['sticky_actions'][9] == 1:  # Shot
                    components["shot_reward"][i] = self.shot_reward_coef

                if o['sticky_actions'][4] == 1:  # Dribbling
                    components["dribble_reward"][i] = self.dribble_reward_coef

                # Position reward based on closeness to the opponent's goal
                opponent_goal_y = (0.0, 1.0)  # Assuming top of the field is opponent's goal
                distance_to_goal = np.linalg.norm(np.array([o['ball'][0], o['ball'][1]]) - np.array(opponent_goal_y))
                components["position_reward"][i] = (1 / (1 + distance_to_goal)) * self.position_reward_coef

            # Summing all components to form the final modified reward
            reward[i] += sum(components[c][i] for c in components)

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, reward_components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        return obs, reward, done, info
