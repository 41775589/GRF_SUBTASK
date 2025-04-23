import gym
import numpy as np
class ClearanceRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on mastering clearance of the ball from defensive zones under pressure,
    ensuring the clearance is both safe and effective."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define clearance zones in the defense area
        self.defensive_zone_threshold = -0.5  # X position in left side of the field
        self.clearance_distance_threshold = 0.4  # Minimum Y distance for effective clearance
        self.clearance_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ClearanceRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        return from_picle

    def reward(self, reward):
        assert len(reward) == len(self.env.unwrapped.observation())
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball' not in o or 'ball_direction' not in o:
                continue

            # Check if ball is in the defensive zone and if it's moving away from the goal
            if o['ball'][0] < self.defensive_zone_threshold and abs(o['ball_direction'][1]) > self.clearance_distance_threshold:
                if o['ball_owned_team'] == 0:  # Assuming 0 is the team that should clear the ball
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += components["clearance_reward"][rew_index]
        
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
