import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for shooting techniques at optimal angles and timing near the goal."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for sticky actions
        self.goal_zone_threshold = 0.2  # Define threshold near opponent's goal considers high-pressure scenario

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            # Check if close to opponent's goal and evaluate ball owning condition
            if o['ball'][0] > 1 - self.goal_zone_threshold and o['ball_owned_team'] == 1:
                optimal_shooting_bonus = 0.1  # Reward for being ready to shoot under pressure

                # Reward if player has a clear trajectory to goal without any players directly in front
                clear_shot = not any([
                    player[0] > o['ball'][0] and abs(player[1] - o['ball'][1]) < 0.1
                    for player in o['left_team']])

                if clear_shot:
                    components['optimal_shooting_bonus'] = [0] * len(reward)
                    components['optimal_shooting_bonus'][idx] += optimal_shooting_bonus
                    reward[idx] += optimal_shooting_bonus

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
