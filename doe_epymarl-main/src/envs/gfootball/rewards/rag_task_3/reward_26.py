import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting accuracy and power reward focused on controlled 'Shot' execution."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_thresholds = {
            'low_power': 0.2,
            'medium_power': 0.5,
            'high_power': 0.8
        }
        self.shot_reward_scaler = 0.15
        self.power_used = [0.0, 0.0]
        self.last_ball_position = np.zeros(3)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.power_used = [0.0, 0.0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'power_used': self.power_used}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.power_used = from_pickle['CheckpointRewardWrapper']['power_used']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_power_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if player performed a shot action
            if self.sticky_actions_counter[0] >= 1:  # Assuming index 0 is 'Shot'
                ball_moving_speed = np.linalg.norm(o['ball_direction'])
                power_used = np.clip((ball_moving_speed - self.last_ball_position), 
                                     0, 1)[0]  # normalize the speed change to get used power estimation
                self.power_used[rew_index] += power_used

                # Determine the power category and apply corresponding reward
                for power_category, threshold in self.shot_thresholds.items():
                    if self.power_used[rew_index] > threshold:
                        components["shooting_power_reward"][rew_index] += self.shot_reward_scaler
                        reward[rew_index] += components["shooting_power_reward"][rew_index]

            self.last_ball_position = np.array(o['ball'])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        
        # Maintain a counter for sticky actions to detect Shot action
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action:
                    self.sticky_actions_counter[i] += 1
                
        return observation, reward, done, info
