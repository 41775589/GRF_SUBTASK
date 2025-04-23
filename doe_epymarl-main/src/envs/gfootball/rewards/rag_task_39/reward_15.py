import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for efficiently clearing the ball from defensive zones under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.clearance_zone_threshold = -0.3  # Threshold to define defensive zone along x-axis
        self.clearance_success_reward = 1.0   # Reward for successfully clearing ball from defensive zone
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = self.env.get_state(to_pickle)
        return state_info

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

        # Checking ball clearance from defensive zones.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball'][0] <= self.clearance_zone_threshold and \
               o['left_team'][rew_index][0] <= self.clearance_zone_threshold and \
               o['ball_owned_team'] == 0 and \
               o['ball_owned_player'] == o['active']:

                # Calculate distance ball was cleared by active player
                if 'ball_direction' in o:
                    ball_clearance = o['ball'][0] + o['ball_direction'][0]
                    if ball_clearance > self.clearance_zone_threshold:
                        components['clearance_reward'][rew_index] = self.clearance_success_reward
                        reward[rew_index] += components['clearance_reward'][rew_index]

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
