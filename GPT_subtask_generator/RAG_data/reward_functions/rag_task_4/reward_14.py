import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for dribbling and the use of sprint in offensive positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribbling_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            # Check for ball possession and sprint usage by the player
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                if o['sticky_actions'][8]:  # action_sprint is the 9th index in sticky_actions
                    components["dribbling_reward"][i] = 0.5
                    reward[i] += components["dribbling_reward"][i]
                    self.sticky_actions_counter[8] += 1  # count sprint action usage

            # Encourage progression towards the opponent's goal (positive x-direction)
            ball_progress = o['ball'][0] - o.get('prev_ball_x', o['ball'][0])
            if ball_progress > 0 and o['ball_owned_team'] == 0:
                progression_reward = ball_progress * 0.2
                components["dribbling_reward"][i] += progression_reward
                reward[i] += components["dribbling_reward"][i]

            # Store current ball x-position for the next step comparison
            o['prev_ball_x'] = o['ball'][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
