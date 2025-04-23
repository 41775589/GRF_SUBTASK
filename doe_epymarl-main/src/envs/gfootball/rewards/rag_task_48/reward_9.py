import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards high passes from midfield, aiming to optimize placement and timing 
    for creating direct scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_reward = {}
        self.high_pass_reward = 0.3
        self.optimal_zone = {"x_min": 0.0, "x_max": 0.4, "y_min": -0.42, "y_max": 0.42}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_reward = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.midfield_reward
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_reward = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Assume midfield zone is between x = 0.0 to 0.4
            # Predicting ball's next location to check if it's a high pass
            ball_location = o['ball']
            ball_direction = o['ball_direction']
            predicted_ball_location = [ball_location[0] + ball_direction[0],
                                       ball_location[1] + ball_direction[1]]

            # If ball is with active player in midfield and the ball movement prediction is in the optimal zone
            ball_in_midfield = 0.0 < ball_location[0] < 0.4
            predicted_location_in_zone = (self.optimal_zone["x_min"] <= predicted_ball_location[0] <= self.optimal_zone["x_max"] and
                                          self.optimal_zone["y_min"] <= predicted_ball_location[1] <= self.optimal_zone["y_max"])

            if ball_in_midfield and predicted_location_in_zone and o['ball_owned_team'] == 0:
                # Reward high passes from midfield
                if rew_index not in self.midfield_reward:
                    self.midfield_reward[rew_index] = True
                    components["high_pass_reward"][rew_index] = self.high_pass_reward

                reward[rew_index] += components["high_pass_reward"][rew_index]

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
