import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for high passes and effective crossings."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.pass_coefficient = 0.3
        self.cross_coefficient = 0.5
        self.ball_height_threshold = 0.15  # Threshold to consider it a high pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.cross_successful = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.cross_successful = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Extract current observations to access ball movement and control variables
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "cross_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Checking ball's height to determine if it's a high pass
            if o['ball'][2] > self.ball_height_threshold:
                components["pass_reward"][rew_index] = self.pass_coefficient
                reward[rew_index] += components["pass_reward"][rew_index]

            # Checking if a cross into the penalty box occurred
            ball_position = np.array(o['ball'][:2])
            penalty_box_bounds = np.array([[0.7, -0.2], [1.0, 0.2]])  # Approximate penalty box front area
            if np.all(ball_position >= penalty_box_bounds[0]) and np.all(ball_position <= penalty_box_bounds[1]):
                if not self.cross_successful:
                    self.cross_successful = True
                    components["cross_reward"][rew_index] += self.cross_coefficient
                    reward[rew_index] += components["cross_reward"][rew_index]

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
