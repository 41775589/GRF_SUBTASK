import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    Adds a dense reward focused on defensive actions, specifically tackling and sliding,
    optimizing response times against direct opponent attacks.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_position = np.array(o['right_team'][o['active']]) if o['active'] < len(o['right_team']) else None
            ball_position = np.array(o['ball'][:2])
            
            # Reward for being close to the ball when defensive actions required:
            if o['game_mode'] == int(libgame.e_GameMode.e_GameMode_Normal) and o['ball_owned_team'] == 1:
                dist_to_ball = np.linalg.norm(ball_position - active_position) if active_position is not None else float('inf')
                
                if dist_to_ball < 0.1: # Very close to ball
                    reward[rew_index] += 0.2

            # Reward for using defensive skills efficiently
            if o['sticky_actions'][6]:  # Choose action_bottom for simulation of sliding action
                reward[rew_index] += 0.5
            elif o['sticky_actions'][0]: # Choose action_left for simulation of tactical repositioning
                reward[rew_index] += 0.1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
