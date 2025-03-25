import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on enhancing central midfield synergy and controlled pace management. 
    This reward function emphasizes successful passes and movement in the midfield area, ensuring pace 
    balance and fluid transitions between different areas of the field.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters for midfield play enhancement
        self.midfield_focus_region = [-0.25, 0.25]  # x-region considered as midfield
        self.pass_reward = 0.3  # Reward for successful passes within midfield
        self.pace_management_reward = 0.1  # Reward for maintaining a balanced pace
        self.pace_threshold = 0.1  # Velocity threshold for controlled pace

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['StickyActionsCounter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['StickyActionsCounter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "pace_management_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Check if any player is in the midfield
            if obs['ball'][0] >= self.midfield_focus_region[0] and obs['ball'][0] <= self.midfield_focus_region[1]:
                # Reward for successful passes within the midfield
                if obs['ball_owned_team'] == 0 and obs['game_mode'] == 0:
                    components['pass_reward'][i] = self.pass_reward
                    reward[i] += components['pass_reward'][i]

                # Reward for controlled ball pace in midfield
                ball_velocity = np.linalg.norm(obs['ball_direction'][:2])
                if ball_velocity < self.pace_threshold:
                    components['pace_management_reward'][i] = self.pace_management_reward
                    reward[i] += components['pace_management_reward'][i]

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
            for j, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[j] = action if action == 1 else self.sticky_actions_counter[j]
        return observation, reward, done, info
