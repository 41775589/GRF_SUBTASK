import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on enhancing mid to long-range passing for strategic team plays."""
    
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
        """Calculate reward with specifications for strategic long passes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        # Ensure reward structure is returned even when no observations
        if observation is None:
            return reward, components

        for idx, (rew, obs) in enumerate(zip(reward, observation)):
            # Basic score reward
            base_reward = rew

            # Rewarding long passes:
            # Criteria for long pass reward:
            # - the pass must cover a significant distance
            # - the pass must change possession state to a teammate who is not adjacent
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:  # assuming agent team is left team (team 0)
                ball_owned_player = obs['ball_owned_player']
                prev_ball_position = obs['ball'] - obs['ball_direction']
                if 0 <= ball_owned_player < len(obs['left_team']):
                    player_x, player_y = obs['left_team'][ball_owned_player]
                    ball_x, ball_y = obs['ball'][:2]
                    dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)

                    if dist > 0.3:  # Considering a significant distance to be more than 30% of field length
                        components["long_pass_reward"][idx] = 0.2  # Reward for good long pass

            reward[idx] = base_reward + components["long_pass_reward"][idx]
        
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
