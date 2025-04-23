import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes from midfield."""

    def __init__(self, env):
        super().__init__(env)
        self.high_pass_effectiveness = 0.2
        self.pass_threshold = 0.2  # Hypothetical threshold to consider a pass a 'high pass'
        self.field_center_x = 0.0  # x-coordinate of midfield
        self.threshold_x_distance = 0.5  # Distance from center to be considered as midfield
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'high_pass_reward': [0.0, 0.0]}

        for i, o in enumerate(observation):
            ball_position = o['ball']
            ball_direction = o['ball_direction']

            # Calculate the elevation angle of the ball's trajectory
            elevation_angle = np.arctan2(ball_direction[2], np.sqrt(ball_direction[0]**2 + ball_direction[1]**2))

            midfield_players = [
                idx for idx, pos in enumerate(o['right_team'] if abs(pos[0] - self.field_center_x) < self.threshold_x_distance)
            ]

            # Reward high passes originating from midfield
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] in midfield_players:
                if elevation_angle > np.radians(self.pass_threshold):
                    # Assuming the pass goes near the opponent's goal area
                    if abs(ball_position[0]) > 1 - self.threshold_x_distance:
                        components['high_pass_reward'][i] += self.high_pass_effectiveness
                        reward[i] += components['high_pass_reward'][i]

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
