import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes from midfield effectively."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To monitor sticky actions usage
        self.high_pass_accuracy_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset sticky actions counter
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            field_area_y = abs(o['ball'][1])
            mid_zone_x = (o['ball'][0] > -0.2) and (o['ball'][0] < 0.2)
            
            # Check if the ball is in midfield zone and high in the air (z > 0.15)
            if mid_zone_x and o['ball'][2] > 0.15:
                if o['ball_owned_team'] == 0:  # 0 indicates left team controls the ball
                    # Check if it was a high pass (approximated by recent direction upwards and sufficient y speed)
                    if (o['ball_direction'][2] > 0.05 and abs(o['ball_direction'][1]) > 0.05):
                        components["high_pass_reward"][rew_index] += self.high_pass_accuracy_reward
                        reward[rew_index] += components["high_pass_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += int(action_active)
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
