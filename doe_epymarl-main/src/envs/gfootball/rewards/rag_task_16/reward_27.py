import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for executing high passes with precision in the
    Google Research Football environment. This focuses on encouraging the agent to learn
    the necessary technical skills for high passes including trajectory control,
    power assessment, and correct situational use.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define specific thresholds or parameters related to ball trajectory and control
        self.high_pass_action = 9  # Assuming index 9 corresponds to "action_high_pass"
        self.distance_threshold = 0.4  # Threshold for ball advancement in y-axis to count as high pass
        self.pass_reward = 0.3  # Reward granted for a successful high pass

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
        # Initial reward and components dictionary to track individual components
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}

        observations = self.env.unwrapped.observation()
        if observations is None:
            return reward, components

        for rew_index in range(len(reward)):
            observation = observations[rew_index]
            current_reward = reward[rew_index]

            # Check if high pass action is performed
            if observation['sticky_actions'][self.high_pass_action]:
                # Calculate the advancement of the ball in the y-axis (higher value for vertical field setups)
                ball_movement_y = observation['ball_direction'][1]

                if abs(ball_movement_y) >= self.distance_threshold:
                    # Adjust the reward based on achieving a notable vertical ball movement
                    components["high_pass_reward"][rew_index] = self.pass_reward
                    current_reward += self.pass_reward

            reward[rew_index] = current_reward

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
