import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a custom reward for executing accurate long passes
    in different zones of the playfield, encouraging vision, timing, and precision
    in ball distribution.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.required_distance_for_long_pass = 0.3  # Threshold for considering a pass as a long pass
        self.reward_for_successful_long_pass = 0.5  # Reward for a successful long pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            current_obs = observation[i]
            ball_pos_init = current_obs.get('ball', None)
            if ball_pos_init is None:
                continue

            # Simulate a step to find the new ball position after action
            _, new_obs, _, _ = self.env.step([None, None])
            ball_pos_final = new_obs[i].get('ball', ball_pos_init)

            # Calculate ball movement distance
            dx = ball_pos_final[0] - ball_pos_init[0]
            dy = ball_pos_final[1] - ball_pos_init[1]
            ball_distance = np.sqrt(dx**2 + dy**2)

            # Check if the ball was moved a long distance
            if ball_distance > self.required_distance_for_long_pass:
                components["long_pass_reward"][i] = self.reward_for_successful_long_pass
                reward[i] += components["long_pass_reward"][i]

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
