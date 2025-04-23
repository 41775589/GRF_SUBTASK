import gym
import numpy as np
class WingerTrainingRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focuses on winger's crossing and sprinting abilities."""

    def __init__(self, env):
        super(WingerTrainingRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._crossing_reward = 1.0
        self._high_speed_dribbling_reward = 0.5
        self._position_reward_scaler = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['WingerTrainingRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['WingerTrainingRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "crossing_reward": [0.0] * len(reward),
            "high_speed_dribbling_reward": [0.0] * len(reward),
            "position_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index in range(len(reward)):
            o = observation[index]
            right_team_roles = o.get('right_team_roles')
            is_winger = right_team_roles[index] in (6, 7)  # Right midfield (RM, LM)

            # Encourage speed and ball control
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == index:
                x_velocity = o['right_team_direction'][index][0]
                y_velocity = np.abs(o['right_team_direction'][index][1])
                
                # Check for high-speed movement in horizontal direction
                if x_velocity > 0.01:
                    components["high_speed_dribbling_reward"][index] = self._high_speed_dribbling_reward

                # Reward crossing from near the sideline
                if np.abs(o['right_team'][index][1]) > 0.4 and x_velocity > 0.02:
                    components["crossing_reward"][index] = self._crossing_reward

            # Additional position-based rewards for wingers
            if is_winger:
                distance_to_sidelines = 1 - np.abs(o['right_team'][index][1])
                components["position_reward"][index] = distance_to_sidelines * self._position_reward_scaler
            
            total_additional_rewards = sum(components[comp][index] for comp in components if comp != "base_score_reward")
            reward[index] += total_additional_rewards
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add component values to info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
