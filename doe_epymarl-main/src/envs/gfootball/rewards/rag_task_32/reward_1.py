import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that encourages crossing and sprinting behaviors for wingers. It assigns
    rewards for accurate crossing from wings and high-speed dribbling.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_reward_coefficient = 0.05  # Reward multiplier for sprinting.
        self.crossing_reward_coefficient = 0.1  # Reward multiplier for successful crosses.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward
        
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": [0.0]*len(reward),
            "crossing_reward": [0.0]*len(reward)
        }

        for i, obs in enumerate(observation):
            controlling_player = obs['active']
            role = obs['left_team_roles' if obs['ball_owned_team'] == 0 else 'right_team_roles'][controlling_player]

            # Encourage sprinting for wingers (index 6 - LM, 7 - RM)
            if role in [6, 7] and obs['sticky_actions'][8]:  # index 8 is 'action_sprint'
                components['sprint_reward'][i] = self.sprint_reward_coefficient

            # Reward successful crosses from wingers
            if role in [6, 7] and obs['game_mode'] in [4] and obs['ball_owned_player'] == controlling_player:  # 4 - Corner kick
                components['crossing_reward'][i] = self.crossing_reward_coefficient

            # Update overall reward including new components
            reward[i] += components['sprint_reward'][i] + components['crossing_reward'][i]

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
