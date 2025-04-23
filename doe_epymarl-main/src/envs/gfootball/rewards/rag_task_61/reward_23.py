import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing team synergy during possession changes by emphasizing 
    precise timing and strategic positioning during both offensive and defensive moves."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.positive_synergy_reward = 1.0
        self.negative_synergy_reward = -0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['synergy_state'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        recovered_state = self.env.set_state(state)
        return recovered_state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        base_rewards = reward.copy()
        reward_components = {'base_score_reward': base_rewards, 'synergy_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, reward_components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            obs = observation[i]
            ball_owned_player_active = obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']
            teammates_close_to_ball = any(np.linalg.norm(team_pos - obs['ball'][:2]) < 0.1 for team_pos in obs['left_team'])

            if ball_owned_player_active and teammates_close_to_ball:
                reward_components['synergy_reward'][i] = self.positive_synergy_reward
            else:
                reward_components['synergy_reward'][i] = self.negative_synergy_reward

            reward[i] += reward_components['synergy_reward'][i]

        return reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action_state in enumerate(agent_obs['sticky_actions']):
                    info[f'sticky_actions_{i}'] = action_state
        return observation, reward, done, info
