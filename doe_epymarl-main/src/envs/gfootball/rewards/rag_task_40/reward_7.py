import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on enhancing defensive strategies by rewarding 
    strategic positioning, successful tackles, and clearance activities that lead to counterattacks.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        """
        Reset the sticky actions counter when the environment is reset.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Retrieve the state of the environment for checkpointing.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment from the checkpoint.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function focusing on defense and transition to counterattacks.
        """
        obs = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 
                      'defensive_bonus': [0.0] * len(reward)}
                      
        game_mode = obs['game_mode']
        ball_owned_team = obs['ball_owned_team']
        
        for idx in range(len(reward)):
            o = obs[idx]
            if game_mode == 3:  # FreeKick, assuming defensive scenario
                if ball_owned_team == 0:  # ball owned by the opposite team
                    components['defensive_bonus'][idx] = 0.2
                    reward[idx] += components['defensive_bonus'][idx]
        
        return reward, components

    def step(self, action):
        """
        Step through the environment applying the reward wrapper.
        """
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return obs, reward, done, info
