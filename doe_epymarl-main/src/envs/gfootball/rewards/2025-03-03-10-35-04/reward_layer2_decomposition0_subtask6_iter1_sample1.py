import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering short pass techniques under defensive pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_success_reward = 1.0  # Reward for successful pass
        self.pressure_intensity_reward = 0.5  # Reward based on proximity of opponent defenders
        
    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Modify the reward to focus on short pass accuracy and dealing with defensive pressure.
        
        :param reward: list of current reward values from the environment.
        :return: tuple with updated rewards and reward components as a dictionary.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "short_pass_reward": [0.0] * len(reward),
            "pressure_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Assuming the observation for individual agent is encapsulated as a single dictionary
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_player = o.get('ball_owned_player', -1)
            ball_owned_team = o.get('ball_owned_team', -1)
            opponents = o['right_team'] if ball_owned_team == 0 else o['left_team']
            own_team = o['left_team'] if ball_owned_team == 0 else o['right_team']
            ball_position = np.array(o['ball'])

            if ball_owned_team == o['active'] and o['game_mode'] == 0:  # Normal play mode
                if ball_owned_player == o['active']:  # Ball owned by active player
                    # Check if it's a pass situation by looking at sticky actions
                    if o['sticky_actions'][9] == 1:  # Assuming index 9 is 'short_pass'
                        # Reward successful pass completion
                        if np.any(np.linalg.norm(own_team - ball_position, axis=1) < 0.1):
                            components["short_pass_reward"][rew_index] += self.pass_success_reward
                            reward[rew_index] += self.pass_success_reward
                        # Reward based on how close the nearest opponent is when making the pass
                        closest_opponent_distance = np.min(np.linalg.norm(opponents - ball_position, axis=1))
                        pressure_reward = max(0, (0.2 - closest_opponent_distance) * self.pressure_intensity_reward)
                        components["pressure_reward"][rew_index] += pressure_reward
                        reward[rew_index] += pressure_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
