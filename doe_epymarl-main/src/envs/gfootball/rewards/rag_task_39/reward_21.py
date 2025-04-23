import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective ball clearance from defensive zones under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 0.8
        self.safety_margin = 0.2  # To consider ball as safely away
        # Define the defensive zone as x <= -0.5 in normalized field coordinates
        self.defensive_zone_x_max = -0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
 
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if in defensive zone and ball is cleared to safe zone
            if o['ball'][0] <= self.defensive_zone_x_max:
                is_cleared = np.abs(o['ball'][0]) > (self.defensive_zone_x_max + self.safety_margin)
                ball_direction_outwards = o['ball_direction'][0] > 0
                if is_cleared and ball_direction_outwards:
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += 1.5 * components["clearance_reward"][rew_index]

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
