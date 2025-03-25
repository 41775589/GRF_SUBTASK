import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Wrapper that adds a dense reward tailored for practicing high passes with precision."""

    def __init__(self, env):
        super().__init__(env)
        # Parameters defining the rewards for high passes
        self.ball_height_threshold = 0.15  # Threshold for the ball height to be considered 'high'
        self.precision_multiplier = 5  # Multiplier for precision of the landing of the pass near the teammate
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for high ball trajectories
            if o['ball'][2] >= self.ball_height_threshold:
                # Identify the target player to be rewarded, typically the teammate in right position.
                target_player_x, target_player_y = self.identify_target_player_position(o)

                # Calculate the distance to the targeted position to determine precision
                precision = np.sqrt((o['ball'][0] - target_player_x) ** 2 + (o['ball'][1] - target_player_y) ** 2)
                precision_reward = max(0, self.precision_multiplier * (1 - precision))

                components["high_pass_reward"][rew_index] = precision_reward
                reward[rew_index] += precision_reward

        return reward, components

    def identify_target_player_position(self, observation):
        # Dummy function: replace it with actual logic to find the target teammate's position
        return 0.5, 0  # Assume an arbitrary position in this example

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
