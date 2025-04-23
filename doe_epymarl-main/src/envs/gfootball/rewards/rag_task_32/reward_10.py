import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for encouraging wingers to perform crosses.
    Specifically, rewards agents for high-speed dribbling and accurate crossing from the wings.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define zones in normalized horizontal pitch coordinates (left is -1, right is 1) for crossing zones
        self.crossing_zones = {
            'left_wing': (-1.0, -0.5),
            'right_wing': (0.5, 1.0)
        }
        self.cross_reward = 0.5  # Reward for making a cross from appropriate zones
        self.sprint_reward = 0.1  # Reward for sprinting in zones

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "cross_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if in left or right wing zones
            if o['active'] >= 0:  # Check if there's a controlled player active
                x_pos = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]
                
                # Sprinting reward
                if o['sticky_actions'][8]:  # index 8 represents sprinting action
                    if self.crossing_zones['left_wing'][0] <= x_pos <= self.crossing_zones['left_wing'][1] or \
                       self.crossing_zones['right_wing'][0] <= x_pos <= self.crossing_zones['right_wing'][1]:
                        components["sprint_reward"][rew_index] = self.sprint_reward
                        reward[rew_index] += components["sprint_reward"][rew_index]

                # Cross reward - assuming cross action is mapped or identifiable (here checked as dribbling in zone)
                if o['sticky_actions'][9] and abs(o['ball'][1] - 0.42) < 0.08:
                    if self.crossing_zones['left_wing'][0] <= x_pos <= self.crossing_zones['left_wing'][1] or \
                       self.crossing_zones['right_wing'][0] <= x_pos <= self.crossing_zones['right_wing'][1]:
                        components["cross_reward"][rew_index] = self.cross_reward
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
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_value
        return observation, reward, done, info
