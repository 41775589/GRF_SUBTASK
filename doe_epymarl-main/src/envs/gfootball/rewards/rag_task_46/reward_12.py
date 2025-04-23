import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances rewards for successful standing tackles, minimizing penalties and effective possession 
    regaining during both normal gameplay and set-piece defensive situations.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the environment and the counters for sticky actions.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Serializes the state including any modifications made by the reward wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Deserializes the state and restores any internal states of the reward wrapper.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Augments the reward based on the quality of tackles and possession recovery.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'tackle_reward': [0.0] * len(reward),
                      'free_kick_penalty': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            # Strategy: Reward standing tackles only in normal play or set-pieces, penalize card incidents.
            if o['game_mode'] in [0, 2, 3, 4, 6]:  # Normal gameplay or set-piece defensive scenarios
                tackled = self.sticky_actions_counter[7]  # action_bottom_left corresponds to a tackle attempt.
                if tackled:
                    components['tackle_reward'][i] = 0.1  # Reward successful tackles.
                    reward[i] += components['tackle_reward'][i]
                # Checking for free kick due to foul
                if 'left_team_yellow_card' in o or 'right_team_yellow_card' in o:
                    if True in o['left_team_yellow_card'] or True in o['right_team_yellow_card']:
                        components['free_kick_penalty'][i] -= 0.2
                        reward[i] += components['free_kick_penalty'][i]

        return reward, components

    def step(self, action):
        """
        Executes a step in the environment, modifies the reward and returns observations and info.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
