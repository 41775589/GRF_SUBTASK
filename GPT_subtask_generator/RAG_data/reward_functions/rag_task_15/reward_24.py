import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on the precision and accuracy of long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_accuracy_factor = 0.2  # Adjusts the weight of successful long passes

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
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": np.zeros(len(reward))}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the ball is possessed by the active player and a long pass action was taken
            if (o['ball_owned_team'] == o['active'] and 
                    'action_bottom_right' in o['sticky_actions'] or 
                    'action_bottom_left' in o['sticky_actions']):
                ball_travel_distance = np.linalg.norm(o['ball_direction'][:2])  # Ignoring z-component
                if ball_travel_distance > 0.5:  # Threshold for considering it a 'long' pass
                    components["long_pass_reward"][rew_index] = self.long_pass_accuracy_factor
                    reward[rew_index] += components["long_pass_reward"][rew_index]

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
