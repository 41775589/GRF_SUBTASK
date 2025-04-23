import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that increases rewards based on dribbling and dynamic positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.dribble_start_bonus = 0.1
        self.dribble_stop_bonus = 0.1
        self.position_change_bonus = 0.05
        self.previous_positions = [None, None]
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_positions = [None, None]
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "position_change_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for agent_idx in range(len(reward)):
            obs = observation[agent_idx]
            if 'sticky_actions' in obs:
                dribble_action = obs['sticky_actions'][9]
                
                # Reward for starting to dribble
                if dribble_action == 1:
                    components['dribble_reward'][agent_idx] = self.dribble_start_bonus
                    reward[agent_idx] += self.dribble_start_bonus
                
                # Reward for stopping dribble
                if self.sticky_actions_counter[9] == 1 and dribble_action == 0:
                    components['dribble_reward'][agent_idx] += self.dribble_stop_bonus
                    reward[agent_idx] += self.dribble_stop_bonus
                
                self.sticky_actions_counter[9] = dribble_action

                # Reward for dynamic positioning: moving significantly
                current_pos = np.array(obs['left_team'] if obs['ball_owned_team'] == 0 else obs['right_team'])[obs['active']]
                if self.previous_positions[agent_idx] is not None:
                    if np.linalg.norm(current_pos - self.previous_positions[agent_idx]) > 0.05:
                        components['position_change_reward'][agent_idx] = self.position_change_bonus
                        reward[agent_idx] += self.position_change_bonus
                
                self.previous_positions[agent_idx] = current_pos
                    
            else:
                self.previous_positions[agent_idx] = None
        
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
