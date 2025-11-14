import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on enhancing the training for sliding tackles and
    defensive positioning against central attacks, including intercepting ground passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for sticky actions
        
        # Custom reward parameters to tweak the importance of defense-related components
        self.slide_tackle_reward = 0.5
        self.position_reward = 0.3
        self.interception_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state = self.env.set_state(state)
        self.sticky_actions_counter = state['CheckpointRewardWrapper']['sticky_actions_counter']
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "slide_tackle_reward": [0.0] * len(reward),
            "position_reward": [0.0] * len(reward),
            "interception_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            active_player_pos = obs['left_team'][obs['active']]
            ball_pos = obs['ball'][:2]  # Consider ball position in x,y coordinates only

            # Reward for controlling the ball in central areas
            if abs(ball_pos[0]) < 0.3:
                components["position_reward"][rew_index] = self.position_reward
                reward[rew_index] += components["position_reward"][rew_index]

            # Encourage slide tackles based on sticky action for sliding and player's proximity to ball
            if obs['sticky_actions'][9] and np.linalg.norm(active_player_pos - ball_pos) < 0.1:
                components["slide_tackle_reward"][rew_index] = self.slide_tackle_reward
                reward[rew_index] += components["slide_tackle_reward"][rew_index]

            # Reward intercepting passes in central area
            if obs['ball_owned_team'] == 1 and (obs['ball'][0] > -0.1 and obs['ball'][0] < 0.1):
                components["interception_reward"][rew_index] = self.interception_reward
                reward[rew_index] += components["interception_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
