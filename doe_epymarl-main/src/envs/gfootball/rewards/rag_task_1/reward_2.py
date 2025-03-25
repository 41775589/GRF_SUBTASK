import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds offensive maneuvers and dynamic adjustment rewards during varied game phases."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.phase_rewards = {
            0: 0.0,  # Normal play
            1: 0.1,  # KickOff
            2: -0.1, # GoalKick
            3: 0.1,  # FreeKick
            4: 0.2,  # Corner
            5: 0.0,  # ThrowIn
            6: 0.5   # Penalty
        }
        self.quick_attack_bonus = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dynamic_game_mode_reward": [0.0] * len(reward),
                      "quick_attack_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for agent_index in range(len(reward)):
            agent_obs = observation[agent_index]
            game_mode = agent_obs['game_mode']
            
            # Game mode dynamic phase rewards
            components["dynamic_game_mode_reward"][agent_index] += self.phase_rewards.get(game_mode, 0.0)
            
            # Quick Attack Bonus: Higher speed with ball possession leads to a bonus
            if agent_obs['ball_owned_team'] == 0:  # assuming '0' is the team index for agent's team
                ball_speed = np.linalg.norm(agent_obs['ball_direction'])
                components["quick_attack_reward"][agent_index] = self.quick_attack_bonus * ball_speed
            
            reward[agent_index] += components["dynamic_game_mode_reward"][agent_index] + components["quick_attack_reward"][agent_index]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
