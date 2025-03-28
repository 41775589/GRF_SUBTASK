import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on offensive strategies, optimizing team coordination and 
    reaction to force openings and defense breaking through passing, position, and shooting."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_power_multiplier = 2.0
        self.passing_bonus = 1.0
        self.positioning_multiplier = 0.5

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
                      "shot_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful shooting actions
            if o['game_mode'] in [2, 6]:  # Shot taken (GoalKick or Penalty)
                components["shot_reward"][rew_index] = self.shot_power_multiplier * 1
                reward[rew_index] += components['shot_reward'][rew_index]

            # Reward for successful passing and team coordination
            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active']:
                previous_player = o['ball_owned_player']
                current_player = o['active']
                if previous_player != current_player:
                    components["pass_reward"][rew_index] = self.passing_bonus
                    reward[rew_index] += components['pass_reward'][rew_index]

            # Reward based on player's positioning advantage
            distance_to_goal = np.linalg.norm([o['left_team'][o['active']][0] + 1, 
                                               o['left_team'][o['active']][1]])
            components["position_reward"][rew_index] = self.positioning_multiplier * (1 - distance_to_goal)
            reward[rew_index] += components['position_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i] > 0
        return observation, reward, done, info
