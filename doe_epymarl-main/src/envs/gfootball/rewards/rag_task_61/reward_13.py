import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for improving team synergy during possession changes,
    emphasizing precise timing and strategic positioning of both offensive and defensive moves.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i in range(len(reward)):
            o = observation[i]
            components_i = components["possession_change_reward"]

            if o['ball_owned_team'] == 1 or o['ball_owned_team'] == 0:
                # Calculate if the possession changed in this step
                if self.prev_ball_owned_team is not None and o['ball_owned_team'] != self.prev_ball_owned_team:
                    if self.prev_ball_owned_team == -1:
                        # If the team just gained possession, reward this change
                        components_i[i] = 0.1  # Starting with a simple fixed reward for gaining possession
                        reward[i] += components_i[i]

            self.prev_ball_owned_team = o['ball_owned_team']
            
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
