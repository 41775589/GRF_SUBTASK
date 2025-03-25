import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds an auxiliary reward for executing high passes with precision.
    The reward function encourages ball air-time management and controlling the ball
    to land accurately in specified zones on the opponent's half.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_zones = [
            (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8),
            (0.8, 1.0), (-0.2, 0.0), (-0.4, -0.2), (-0.6, -0.4),
            (-0.8, -0.6), (-1.0, -0.8)  # Zones in y-coordinate on the opponent's half
        ]
        self.zone_rewards = np.linspace(0.1, 1.0, len(self.high_pass_zones))
        self.passes_in_zone = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_in_zone = {zone: False for zone in self.high_pass_zones}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.passes_in_zone
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passes_in_zone = from_pickle['CheckpointRewardWrapper']
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
            ball_pos = np.array(o['ball'])  # [x, y, z]
            
            if ball_pos[2] > 0.1:  # the ball is in the air
                for idx, (lower_bound, upper_bound) in enumerate(self.high_pass_zones):
                    if lower_bound <= ball_pos[1] <= upper_bound:  # ball is in a y-zone
                        if not self.passes_in_zone[(lower_bound, upper_bound)]:  # first time in this zone
                            components["high_pass_reward"][rew_index] += self.zone_rewards[idx]
                            self.passes_in_zone[(lower_bound, upper_bound)] = True
                            break

        # Aggregate rewards
        for idx in range(len(reward)):
            reward[idx] += components["high_pass_reward"][idx]

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
