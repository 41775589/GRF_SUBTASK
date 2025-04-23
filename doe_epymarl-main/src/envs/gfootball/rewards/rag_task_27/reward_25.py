import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that emphasizes defensive play by rewarding interception and proper positioning.
    Introduces additional reward components for intercepting opponent passes and maintaining good defensive positioning.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.2
        self.positioning_reward = 0.1

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
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Handling game mode scenarios for positioning
            if o['game_mode'] in (2, 3, 4, 5, 6):  # Non-normal game modes
                reward[rew_index] += o['ball_owned_team'] == 0 * self.positioning_reward
                components["positioning_reward"][rew_index] = self.positioning_reward

            # Reward for interception: ball was owned and now it's not, close to player
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and 
                'previous_ball_owner' in o and o['previous_ball_owner'] == 1):
                dist_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']])  # Only consider x, y
                if dist_to_ball < 0.1:  # Arbitrary distance of 'closeness'
                    reward[rew_index] += self.interception_reward
                    components["interception_reward"][rew_index] = self.interception_reward
                    
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
