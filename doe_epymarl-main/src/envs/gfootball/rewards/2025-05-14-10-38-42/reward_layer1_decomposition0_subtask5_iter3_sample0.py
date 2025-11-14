import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for mastering passes from defensive positions,
    particularly focusing on 'Short Pass' and 'High Pass' under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.short_pass_reward = 0.3  # Reward multiplier for short passes
        self.high_pass_reward = 0.5  # Reward multiplier for high passes
        self.defensive_zone_threshold = -0.3  # Defensive zone threshold
        self.pressure_threshold = 1.2  # Distance threshold for considering opponents applying pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "short_pass_reward": [0.0], "high_pass_reward": [0.0]}

        if observation is None:
            return reward, components

        o = observation[0]  # Single agent scenario

        # Check if the agent has the ball and is in the defensive zone
        if o['ball_owned_team'] == 0 and o['left_team'][o['active']][0] < self.defensive_zone_threshold:
            ball_pos = o['ball'][:2]
            player_pos = o['left_team'][o['active']]

            # Calculate pressure based on proximity of any opponent
            opponents_pos = o['right_team']
            pressure = np.any([np.linalg.norm(player_pos - opp) < self.pressure_threshold for opp in opponents_pos])

            if pressure:
                # Check for 'short pass' action (index 6) being active in sticky actions
                if o['sticky_actions'][6]:
                    components["short_pass_reward"][0] = self.short_pass_reward
                    reward[0] += components["short_pass_reward"][0]

                # Check for 'high pass' action (index 7) being active in sticky actions
                if o['sticky_actions'][7]:
                    components["high_pass_reward"][0] = self.high_pass_reward
                    reward[0] += components["high_pass_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
