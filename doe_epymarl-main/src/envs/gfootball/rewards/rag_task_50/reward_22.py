import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for making successful long passes between specific zones on the field.
    This reward function focuses on improving the agents' ability to make accurate long passes, 
    which require excellent vision, timing, and precision.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define zones as pairs of coordinates indicating 'from' and 'to' zones for rewarding successful long passes
        self.pass_zones = [
            ([-1.0, 0], [-0.3, 0.42]), # from left defense to midfield
            ([-0.3, 0], [0.3, 0.42]),  # from midfield to opposite midfield
            ([0.3, 0], [1.0, 0.42])    # from midfield to right forward
        ]
        self.long_pass_threshold = 0.5  # The minimum distance to consider a pass as a long pass
        self.pass_success_reward = 0.2  # Reward added for successful long pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        if observation is None:
            return reward, {"base_score_reward": reward.copy()}

        components = {"base_score_reward": reward.copy(), 
                      "long_pass_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            ball_start = o.get('ball')
            ball_end = o.get('ball') + o.get('ball_direction')

            if o['ball_owned_team'] in (0, 1):  # Check if one of the teams own the ball
                starting_zone = next((zone for zone in self.pass_zones if self._point_in_zone(ball_start[:2], zone[0])), None)
                ending_zone = next((zone for zone in self.pass_zones if self._point_in_zone(ball_end[:2], zone[1])), None)

                # Validate the long pass between predefined zones
                if starting_zone and ending_zone and starting_zone != ending_zone:
                    pass_distance = np.linalg.norm(ball_end[:2] - ball_start[:2])
                    if pass_distance >= self.long_pass_threshold:
                        components["long_pass_reward"][rew_index] += self.pass_success_reward
                        reward[rew_index] += components["long_pass_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info

    def _point_in_zone(self, point, zone):
        """
        Helper function to check if a point is within a given rectangular zone.
        `point` is in form [x, y], and `zone` is in the form [x range, y range].
        """
        x, y = point
        x_min, y_min = zone[0]
        x_max, y_max = zone[1]
        return x_min <= x <= x_max and y_min <= y <= y_max
