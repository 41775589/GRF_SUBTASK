import gym
import numpy as np
class HighPassSkillEnhancementRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for practicing high passes with required precision and power assessment."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for a 'high pass'
        self.min_z_position = 0.15  # Assume this to be the threshold for "high" passes
        self.power_threshold = 0.8  # Assume this to be the threshold for "sufficient power"
        self.pass_reach_radius = 0.1  # Margin around receiver for defining precision

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['HighPassSkillEnhancementRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['HighPassSkillEnhancementRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check for a high pass (ball_position_z higher than threshold) and power measurement
            ball_z_position = o['ball'][2]  
            if ball_z_position > self.min_z_position:
                ball_owned_team = o['ball_owned_team']
                ball_owned_player = o['ball_owned_player']

                if ball_owned_team in [0, 1]:  # Ball is owned by a team (0 = left, 1 = right)
                    receiver_x, receiver_y = self.predict_receiver_position(o)

                    # Evaluating the pass precision by checking if the receiver is within the designated area
                    if (np.abs(receiver_x - o['ball'][0]) <= self.pass_reach_radius and 
                        np.abs(receiver_y - o['ball'][1]) <= self.pass_reach_radius):
                        components["high_pass_reward"][rew_index] = 1.0

            # Calculate total reward
            reward[rew_index] += 2.0 * components["high_pass_reward"][rew_index]

        return reward, components

    def predict_receiver_position(self, observation):
        """ Predict potential receiver's position. This function is a stub and should ideally be implemented 
        based on the dynamics of player positions and ball trajectory."""
        # Example of very simplistic prediction, this should be more sophisticated
        receiver_index = observation['designated']
        if observation['ball_owned_team'] == 0:
            team_positions = observation['left_team']
        else:
            team_positions = observation['right_team']

        receiver_x, receiver_y = team_positions[receiver_index]
        return receiver_x, receiver_y

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
