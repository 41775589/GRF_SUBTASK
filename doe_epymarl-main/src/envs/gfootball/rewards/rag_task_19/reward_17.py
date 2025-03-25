import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for defense and strategic midfield management."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Hyperparameters to weight different components
        self.defense_reward_coefficient = 1.0
        self.midfield_strategy_coefficient = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Getting the raw observations for analysis
        components = {"base_score_reward": reward.copy(), "defense_reward": [0.0] * len(reward),
                      "midfield_strategy_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Extracting specific parts of observations to calculate defense and midfield rewards
            ball_position = o['ball'][0]  # X position of ball
            active_player_position_x = o['right_team'][o['active']][0] if o['designated'] else o['left_team'][o['active']][0]
            
            # Defense Reward: Active player being in a defending position when opponent has the ball
            if o['ball_owned_team'] == 1 and o['ball'][0] < 0:  # Opponents have the ball in the agent's half
                components["defense_reward"][rew_index] = self.defense_reward_coefficient * (0.5 - abs(ball_position + active_player_position_x))
            
            # Midfield Strategy Reward: Assesses the position and control in midfield areas
            if abs(ball_position) < 0.5:  # Ball is in midfield
                components["midfield_strategy_reward"][rew_index] = self.midfield_strategy_coefficient * (0.5 - abs(ball_position))
            
            # Calculating total reward by combining base reward with additional components
            reward[rew_index] += components["defense_reward"][rew_index] + components["midfield_strategy_reward"][rew_index]
        
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

        return observation, reward, done, info
