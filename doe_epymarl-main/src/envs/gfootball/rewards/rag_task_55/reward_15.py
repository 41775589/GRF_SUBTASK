import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful tackles without fouling.
    This wrapper focuses on defensive actions, specifically standing and sliding tackles, and adjusting 
    the reward based on the context (e.g., successful tackle vs. foul).
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
                      "tackle_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            components["tackle_reward"][rew_index] = 0

            # Check if tackle action was applied
            if 'sticky_actions' in o and (o['sticky_actions'][7] == 1 or o['sticky_actions'][6] == 1):
                own_team = 0 if 'left_team_active' in o and o['active'] in o['left_team_active'] else 1
                if 'ball_owned_team' in o and o['ball_owned_team'] == 1 - own_team:
                    # Reward successful tackles based on the context, ensuring no fouls (game_mode != 4 means no foul)
                    if o['game_mode'] == 0:
                        components["tackle_reward"][rew_index] = 0.5
                else:
                    # Penalize fouls
                    if o['game_mode'] == 4:
                        components["tackle_reward"][rew_index] = -0.5

            reward[rew_index] += components["tackle_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
