import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for goalkeeper coordination in high-pressure scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the boundaries within the goalie area considered critical for action
        self.critical_zone_left = [-1.0, -0.3, -1.0, 0.3]  # x_min, y_min, x_max, y_max
        self.critical_zone_right = [1.0, -0.3, 1.0, 0.3]
        self.critical_action_reward = 0.2  # Reward for clearing ball from critical zone
        self.last_ball_position = None

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.last_ball_position = None
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_action_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
           
            ball_position = o["ball"][:2]  # x, y
            game_mode = o["game_mode"]
            ball_owned_team = o["ball_owned_team"]

            # Check critical situations on the left goal
            if game_mode == 0 and ball_owned_team == 1:
                if (self.critical_zone_left[0] <= ball_position[0] <= self.critical_zone_left[2] and
                    self.critical_zone_left[1] <= ball_position[1] <= self.critical_zone_left[3]):
                    # Ensure there was movement out of the critical zone by the goalkeeper
                    if self.last_ball_position is not None and np.linalg.norm(ball_position - self.last_ball_position) > 0.1:
                        components["goalkeeper_action_reward"][rew_index] = self.critical_action_reward
                        reward[rew_index] += components["goalkeeper_action_reward"][rew_index]

            # Check critical situations on the right goal
            elif game_mode == 0 and ball_owned_team == 0:
                if (self.critical_zone_right[0] <= ball_position[0] <= self.critical_zone_right[2] and
                    self.critical_zone_right[1] <= ball_position[1] <= self.critical_zone_right[3]):
                    if self.last_ball_position is not None and np.linalg.norm(ball_position - self.last_ball_position) > 0.1:
                        components["goalkeeper_action_reward"][rew_index] = self.critical_action_reward
                        reward[rew_index] += components["goalkeeper_action_reward"][rew_index]

            self.last_ball_position = np.array(ball_position)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions count
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
