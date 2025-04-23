import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for making accurate long passes between predefined areas on the field.
    This aims to promote development of agents capable of executing long passes with precision and timing.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints for long passes on the field
        self.pass_checkpoints = [
            ([-1, -0.5], [-0.5, 0.5]),  # from left half to middle horizontally
            ([-0.5, 0.5], [0.5, 1]),    # from middle to right half
            ([-1, -0.5], [0.5, 1]),     # from left half directly to right half
            ([0.5, 1], [-1, -0.5])      # from right half directly to left half
        ]
        self.pass_accuracy_reward = 0.2  # Additional reward for completing a good long pass

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

        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]

            for checkpoint in self.pass_checkpoints:
                orig_area, dest_area = checkpoint
                if self.check_in_zone(ball_pos, orig_area) and self.action_matches_pass(o):
                    ball_final_dest = self.env.unwrapped.expected_ball_position_after_actions()
                    if self.check_in_zone(ball_final_dest, dest_area):
                        components["pass_accuracy_reward"][rew_index] = self.pass_accuracy_reward
                        reward[rew_index] += self.pass_accuracy_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def check_in_zone(self, position, zone_bounds):
        """
        Check if a position is within the specified bounds.
        """
        x, y = position
        x_min, x_max = zone_bounds
        return x_min <= x <= x_max

    def action_matches_pass(self, observation):
        """
        Check if the last action was a pass (long pass particularly in context).
        This is a pseudo implementation as exact action checks depend on the environment specifics.
        """
        return observation['sticky_actions'][9] == 1  # Assuming index 9 in sticky_actions is 'action_long_pass'
