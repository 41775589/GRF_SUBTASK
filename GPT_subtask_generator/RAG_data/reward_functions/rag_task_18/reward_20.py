import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining possession and managing controlled pacing in the midfield."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['Middleware Control'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['Middleware Control']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # We focus on midfield control, hence we investigate if the ball is controlled by midfielders
            if 'left_team_roles' in o and 'right_team_roles' in o:
                midfield_roles = [4, 5, 6]  # DM, CM, LM as per role indices
                if o['ball_owned_team'] == 0:  # Assuming 0 for left team, 1 for right team
                    if o['left_team_roles'][o['active']] in midfield_roles:
                        components["midfield_control_bonus"][rew_index] = self.midfield_control_reward
                elif o['ball_owned_team'] == 1:
                    if o['right_team_roles'][o['active']] in midfield_roles:
                        components["midfield_control_bonus"][rew_index] = self.midfield_control_reward

            # Update the reward
            reward[rew_index] += components["midfield_control_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_rewards, components = self.reward(reward)
        info["final_reward"] = sum(new_rewards)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky action info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, new_rewards, done, info
