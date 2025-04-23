import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for effective ball clearances under pressure in defensive zones."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 0.5  # Adjust the reward magnitude for clearances as needed.
        self.pressure_threshold = 0.2  # Distance threshold to consider an opponent applying pressure.
        self.defensive_zone_threshold = -0.5  # X Coordinate threshold to define the defensive zone.

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
        components = {
            "base_score_reward": reward.copy(),
            "clearance_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the ball is in the defensive zone and the team owns the ball.
            if (o['ball'][0] <= self.defensive_zone_threshold and
                    o['ball_owned_team'] == 0 and
                    any(np.linalg.norm(o['right_team'][:,:2] - o['ball'][:2], axis=1) < self.pressure_threshold)):
                # Reward the clearance action
                if 'action' in o and o['action'] == 'clear':
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += self.clearance_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
