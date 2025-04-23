import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for executing high passes from midfield efficiently towards
    creating direct scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.high_pass_effectivity_score = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['high_pass_effectivity_score'] = self.high_pass_effectivity_score
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.high_pass_effectivity_score = from_pickle.get('high_pass_effectivity_score', 0.5)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Require length matches between rewards and observations
        assert len(reward) == len(observation)

        # Evaluate each agent based on the task expectation
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Checking if a high pass is likely to be made from midfield.
            # Assume midfield y-range is approximately -0.2 to 0.2 in normalized coords.
            midfield_y_range = (-0.2, 0.2)
            ball_midfield = midfield_y_range[0] <= o['ball'][1] <= midfield_y_range[1]

            # Check if a successful pass occurred and was directed forwards.
            # Sticky action 9 (array index) corresponds to high pass as hypothetically mapped.
            # checking ball_direction towards forward (parabolic trajectory with z > 0 is assumed for simplicity).
            if ball_midfield and o['sticky_actions'][9] and o['ball'][2] > 0.1:
                components["high_pass_reward"][rew_index] = self.high_pass_effectivity_score
                reward[rew_index] += self.high_pass_effectivity_score

        return reward, components

    def step(self, action):
        # Step through environment with the current action
        observation, reward, done, info = self.env.step(action)
        
        # Compute custom reward and append component values
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
