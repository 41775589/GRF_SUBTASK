import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive actions and ball interception."""

    def __init__(self, env):
        super().__init__(env)
        self.baseline_tiredness = 0.3  # Initial assumption of players being less tired
        self.encouragement_for_defense = 1.5  # Coefficient for defensive actions
        self.intercept_reward = 2.0  # Reward for intercepting the ball
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positioning_reward = 0.1  # Additional reward for good defensive positioning
        self.last_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward agents for being in a good defensive position and not being too tired
            if o['left_team_active'][o['active']] and o['left_team_tired_factor'][o['active']] < self.baseline_tiredness:
                defense_x_position = min(p[0] for p in o['left_team'])  # Consider good defensive positioning as being back
                if defense_x_position < 0:  # In own half primarily
                    components["defensive_reward"][rew_index] += self.defensive_positioning_reward

            # Reward for intercepting the ball (changing ball possession in favor of player's team)
            if self.last_ball_position is not None and self.last_ball_position[0] != o['ball_owned_team'] == 0:
                if o['ball_owned_player'] == o['active']:
                    components["defensive_reward"][rew_index] += self.intercept_reward

            reward[rew_index] += components["defensive_reward"][rew_index]

        # Update the last ball owner team after processing all rewards
        self.last_ball_position = (o['ball_owned_team'], o['ball_owned_player'])

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        return from_pickle

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
