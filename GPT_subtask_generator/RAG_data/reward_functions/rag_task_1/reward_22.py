import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dynamics and quick attack-oriented rewards based on game phases."""
    
    def __init__(self, env):
        super().__init__(env)
        self.quick_attack_bonus = 0.3
        self.dynamic_adaptation_bonus = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        return self.env.set_state(state)
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "quick_attack_bonus": [0.0] * len(reward),
                      "dynamic_adaptation_bonus": [0.0] * len(reward)}

        for agent_idx, (rew, obs) in enumerate(zip(reward, observation)):
            # Encourage fast forward movements in Normal game mode
            if obs['game_mode'] == 0 and obs['ball_owned_team'] == 1:  # Own team has the ball
                ball_speed = np.linalg.norm(obs['ball_direction'][:2])
                if ball_speed > 0.01:  # Ball is moving forward quickly
                    components["quick_attack_bonus"][agent_idx] = self.quick_attack_bonus
                    rew += components["quick_attack_bonus"][agent_idx]
            
            # Adaptive response: reward for changing strategy effectively in corner or freekick situations
            if obs['game_mode'] in [3, 4]:  # Corner or free-kick
                # Check if recent actions have led to gaining ball possession or setting up a strategic position
                if obs['ball_owned_team'] == 1:
                    components["dynamic_adaptation_bonus"][agent_idx] = self.dynamic_adaptation_bonus
                    rew += components["dynamic_adaptation_bonus"][agent_idx]
            
            # Update overall reward
            reward[agent_idx] = rew

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
