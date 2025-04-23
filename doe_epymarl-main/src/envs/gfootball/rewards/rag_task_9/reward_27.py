import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for offensive football skills such as passing, shooting, and dribbling.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coefficients for different actions
        self.pass_reward_coeff = 0.03
        self.shot_reward_coeff = 0.07
        self.dribble_reward_coeff = 0.05
        self.sprint_reward_coeff = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Enhance the base reward based on offensive actions performed by agents.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward).copy(), 
                      "pass_reward": np.zeros_like(reward),
                      "shot_reward": np.zeros_like(reward),
                      "dribble_reward": np.zeros_like(reward),
                      "sprint_reward": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        action_indices = {
            'Short Pass': 7,
            'Long Pass': 8,
            'Shot': 9,
            'Dribble': 1,
            'Sprint': 6
        }

        for i, obs in enumerate(observation):
            sticky_actions = obs['sticky_actions']
            
            # Reward for successful pass (short or long)
            if sticky_actions[action_indices['Short Pass']] or sticky_actions[action_indices['Long Pass']]:
                components['pass_reward'][i] = self.pass_reward_coeff
                reward[i] += components['pass_reward'][i]
            
            # Reward for attempting a shot
            if sticky_actions[action_indices['Shot']]:
                components['shot_reward'][i] = self.shot_reward_coeff
                reward[i] += components['shot_reward'][i]

            # Reward for dribbling
            if sticky_actions[action_indices['Dribble']]:
                components['dribble_reward'][i] = self.dribble_reward_coeff
                reward[i] += components['dribble_reward'][i]

            # Reward for sprinting effectively
            if sticky_actions[action_indices['Sprint']]:
                components['sprint_reward'][i] = self.sprint_reward_coeff
                reward[i] += components['sprint_reward'][i]

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
