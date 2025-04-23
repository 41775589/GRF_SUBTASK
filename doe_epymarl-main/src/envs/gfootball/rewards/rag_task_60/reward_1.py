import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defense-oriented behaviors in the Google Research Football environment."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Base reward components setup
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_efficiency": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            if o is None:
                continue
            
            # Defensive behavior incentives:
            # Encourage maintaining positions, interpreting player velocities and stopping ball progression
            # Calculate velocity norm for active players
            player_velocity = np.linalg.norm(o['left_team_direction'][o['active']])
            ball_approaching = np.linalg.norm(o['ball_direction']) > 0.01
            
            # Reward rapid deceleration if the ball is approaching
            if player_velocity < 0.01 and ball_approaching:
                components["defense_efficiency"][i] = 0.05
                reward[i] += components["defense_efficiency"][i]
            
            # Reward if player is in a strategic position when the opposite team controls the ball
            ball_owned_by_opponent = (o['ball_owned_team'] == 1)
            strategic_position = abs(o['left_team'][o['active']][0]) < 0.1  # High X values near the center are strategic
            if ball_owned_by_opponent and strategic_position:
                components["defense_efficiency"][i] += 0.1
                reward[i] += components["defense_efficiency"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["total_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
