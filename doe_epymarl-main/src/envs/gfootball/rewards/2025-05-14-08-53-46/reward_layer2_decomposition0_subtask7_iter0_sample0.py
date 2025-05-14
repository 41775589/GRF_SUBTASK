import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for positioning and movement to develop defensive understanding.
    The subtasks involve positioning to cut off passes and supporting fellow defenders.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_coefficient = 0.1

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
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "positional_reward": [0.0]}

        for agent_index in range(len(reward)):  # assumed for single or multi-agent scenarios
            o = observation[agent_index]
            
            # Defensive positioning reward based on distance from the ball and nearness to other defenders
            if o['ball_owned_team'] == 1:  # Assuming team '0' is our team
                distance_to_ball = np.linalg.norm(o['left_team'][o['active']] - o['ball'][:2])
                mean_distance_to_defenders = np.mean([np.linalg.norm(o['left_team'][o['active']] - teamate) 
                                                       for i, teamate in enumerate(o['left_team']) if i != o['active']])
                components["positional_reward"][agent_index] = self.positional_coefficient * (1.0 / (1.0 + distance_to_ball)
                                                                                             + 1.0 / (1.0 + mean_distance_to_defenders))
                
                # Updating the reward for the agent
                reward[agent_index] += components["positional_reward"][agent_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # Summing reward if multi-agent to single value
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
