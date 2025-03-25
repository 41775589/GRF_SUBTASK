import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on the precision and distance of long passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.pass_target_zones = np.asarray([
            [0.0, 0.8],  # Zone 1: Long pass towards the attacking third
            [0.8, 1.0],  # Zone 2: Target zone near the opponent's goal area
        ])
        self.pass_coefficients = [1.0, 2.0]  # Reward coefficients for each zone
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            if o['ball_owned_team'] == 0:  # Assuming 0 is the controlled team
                ball_position = o['ball'][0]
                ball_destination = o.get('ball_direction', [0, 0])[0] + ball_position
                for j, zone in enumerate(self.pass_target_zones):
                    if zone[0] <= ball_destination <= zone[1]:
                        additional_reward = self.pass_coefficients[j] * (ball_destination - ball_position)
                        components["long_pass_reward"][i] += additional_reward
                        reward[i] += components["long_pass_reward"][i]
                        break

        return reward, components

    def step(self, action):
        observation, original_reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(original_reward)
        info["final_reward"] = sum(modified_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for index, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[index] += 1
        return observation, modified_reward, done, info
