import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful standing tackles in normal gameplay,
    enhancing agents' ability to regain possession without committing fouls.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_successful = 0
        self.tackle_penalty_coef = -0.5
        self.tackle_success_coef = 1.0
        self.minimum_tackles = 1

    def reset(self):
        self.tackles_successful = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.tackles_successful
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackles_successful = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_reward = reward.copy()
        tackle_reward = [0.0] * len(reward)
        
        for i, obs in enumerate(observation):
            # Check if there was a tackle
            if obs['game_mode'] in [3, 4]:  # FreeKick or Corner, where tackles often occur
                if 'ball_owned_team' in obs and obs['ball_owned_team'] != self.env.unwrapped.team_to_retrieve_ball:
                    # Tackle seemed effective in regaining possession
                    self.tackles_successful += 1
                    tackle_reward[i] = self.tackle_success_coef
                else:
                    # Tackle did not regain possession or caused a foul
                    tackle_reward[i] = self.tackle_penalty_coef
            
            reward[i] = base_reward[i] + tackle_reward[i]
        
        return reward, {"base_score_reward": base_reward, "tackle_reward": tackle_reward}

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
