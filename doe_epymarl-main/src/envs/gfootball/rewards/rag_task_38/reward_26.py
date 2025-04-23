import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that encourages and rewards initiating counterattacks through accurate long passes
    and quick transitions from defense to attack.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_efficiency_reward = 0.2  # Reward coefficient for successful long passes
        self.transition_speed_bonus = 0.1   # Reward coefficient for quick transition

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(state_info)

    def set_state(self, state):
        more_info = self.env.set_state(state)
        self.sticky_actions_counter = more_info['sticky_actions_counter']
        return more_info

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_efficiency_reward": [0.0] * len(reward), 
                      "transition_speed_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            ball_speed = np.linalg.norm(obs['ball_direction'])
            distance_covered = np.linalg.norm(obs['ball'][:2])
            if obs['game_mode'] == 2:  # Counter checking for potential game mode indicating long pass
                components["pass_efficiency_reward"][rew_index] = self.pass_efficiency_reward
            # Checking for rapid transitions
            if ball_speed > 0.03 and distance_covered > 0.5:
                components["transition_speed_bonus"][rew_index] = self.transition_speed_bonus
            
            # Computing the final modified reward for this team
            reward[rew_index] += components["pass_efficiency_reward"][rew_index] + components["transition_speed_bonus"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
