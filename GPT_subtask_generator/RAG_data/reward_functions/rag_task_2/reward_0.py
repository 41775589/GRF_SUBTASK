import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on defensive strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reset()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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
        components = {"base_score_reward": reward.copy(), "defensive_positioning": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 1 and o['active'] != -1:
                x, y = o['right_team'][o['active']]
                ball_x, ball_y = o['ball'][:2]
                distance_to_ball = np.sqrt((x - ball_x) ** 2 + (y - ball_y) ** 2)

                # Reward for properly positioning in relation to the ball while on defensive duty
                if distance_to_ball < 0.1:
                    components["defensive_positioning"][rew_index] = 0.05
                elif distance_to_ball < 0.2:
                    components["defensive_positioning"][rew_index] = 0.03
                elif distance_to_ball < 0.3:
                    components["defensive_positioning"][rew_index] = 0.01

                reward[rew_index] += components["defensive_positioning"][rew_index]

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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
