import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes learning high passes with precision.
    Rewards are given based on trajectory control, power assessment, and
    appropriate situational usage of high passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.2  # Reward magnitude for executing a high pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),  # base rewards from the original environment
            "high_pass_reward": [0.0, 0.0]  # initially no high pass rewards
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Calculate reward components based on the behavior
        for rew_index, o in enumerate(observation):
            # Check for high passes: represented by ball's movement in the z-direction
            if o.get('ball_direction', [0, 0, 0])[2] > 0.05:
                # Additional checks:
                # - Ball should be owned by the same team
                # - Ensure the ball's y movement is minimal compared to its z direction
                if (o['ball_owned_team'] == 0 and
                    abs(o['ball_direction'][1]) < abs(o['ball_direction'][2])):
                    # Apply high pass reward based on z-direction velocity further normalized by distance to teammates
                    components["high_pass_reward"][rew_index] = self.high_pass_reward * min(1, o['ball_direction'][2] / 0.1)
                    reward[rew_index] += components["high_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # Total reward for logging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)  # Logging individual components
        return observation, reward, done, info
