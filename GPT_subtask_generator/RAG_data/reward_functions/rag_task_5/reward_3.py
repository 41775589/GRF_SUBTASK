import gym
import numpy as np
class DefensiveTrainingRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive actions and counter-attack potentials."""
    def __init__(self, env):
        super(DefensiveTrainingRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['DefensiveTrainingRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for having defensive positioning
            if o['ball_owned_team'] == 1:  # If the opponent has the ball
                distance_from_goal = np.linalg.norm(o['left_team'][o['designated']] - np.array([-1, 0]))
                defensive_factor = max(0, (0.3 - distance_from_goal))  # Closer to goal, higher reward
                components["defensive_reward"][rew_index] = defensive_factor
                
            # Reward for regaining possession, indicating an immediate counter-attack opportunity
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["defensive_reward"][rew_index] += 0.2  # Reward for taking possession

            # Aggregate the rewards
            reward[rew_index] += components["defensive_reward"][rew_index]

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
