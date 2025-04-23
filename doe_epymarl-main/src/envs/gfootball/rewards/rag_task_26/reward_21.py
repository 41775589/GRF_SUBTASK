import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specifically tailored rewards for mastering midfield dynamics, 
    focusing on players' roles and effective control in the midfield area."""
    
    def __init__(self, env):
        super().__init__(env)
        self.midfield_control_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}

        for idx in range(len(reward)):
            obs = observation[idx]
            if obs is None:
                continue
            
            # Reward midfield control based on role and position
            team_roles = obs['left_team_roles'] if obs['designated'] == obs['active'] else obs['right_team_roles']
            team_position = obs['left_team'] if obs['designated'] == obs['active'] else obs['right_team']
            role = team_roles[obs['active']]
            pos_x, pos_y = team_position[obs['active']]

            # Central Midfields (CM) and Wide Midfields (LM, RM) are pivotal
            if role in [5, 6, 7]:  # CM, LM, RM
                # Central midfielders or wide midfielders near the center get additional control rewards
                if abs(pos_y) < 0.3:  # Closer to center line
                    control_bonus = 1.0 if np.linalg.norm(obs['ball'][:2]) < 0.5 else 0.5
                    components["midfield_control_reward"][idx] = self.midfield_control_reward * control_bonus
                    reward[idx] += components["midfield_control_reward"][idx]

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
