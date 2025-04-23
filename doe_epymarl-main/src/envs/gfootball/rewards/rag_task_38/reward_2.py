import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for initiating counterattacks with long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counterattack_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            base = reward[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'ball_owned_player' in o:
                ball_change_factor = np.linalg.norm(o['ball_direction'])
                player_pos = o['right_team'][o['ball_owned_player']]
                ball_pos = o['ball']
                
                # Reward for controlling the ball in own half and making a forward long pass
                if player_pos[0] < 0 and ball_pos[0] > player_pos[0]:
                    distance = np.linalg.norm([ball_pos[1] - player_pos[1]])
                    pass_quality = min(distance, 1)  # Normalize the distance for reward calculation
                    components["counterattack_reward"][rew_index] = 0.5 * pass_quality
                
                # Encourage quick transition from defense to attack
                if (player_pos[0] - o['ball'][0]) ** 2 < 0.1:  # Detect sudden change in possession in defense zone
                    components["counterattack_reward"][rew_index] += 0.3  # Moderate reward for quick transition initiations
            
            # Aggregate the rewards
            reward[rew_index] = base + components["counterattack_reward"][rew_index]
        
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
            
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
