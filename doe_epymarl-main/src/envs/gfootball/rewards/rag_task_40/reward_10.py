import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper designed to enhance the defensive unit's capability to handle
    direct attacks through improved specialization in confrontational defense
    and strategic positioning for counterattacks.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defender_positions_x = 0.25  # Threshold for defenders' closer to the goal half
        self.confrontational_defense_reward = 0.2
        self.counterattack_opportunity_reward = 0.3

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
                      "confrontational_defense_reward": [0.0] * len(reward),
                      "counterattack_opportunity_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            active_player_x = o['left_team'][o['active']][0]  # X position of the controlled player
            ball_x = o['ball'][0]

            # Reward for moving into defensive positions when opponent possesses the ball
            if o['ball_owned_team'] == 1 and active_player_x < -self.defender_positions_x:
                components["confrontational_defense_reward"][rew_index] = self.confrontational_defense_reward
                reward[rew_index] += components["confrontational_defense_reward"][rew_index]
            
            # Reward for moving into position to initiate a counterattack when possession changes to left team
            if o['ball_owned_team'] == 0 and o['ball'][0] < 0 and active_player_x > self.defender_positions_x:
                components["counterattack_opportunity_reward"][rew_index] = self.counterattack_opportunity_reward
                reward[rew_index] += components["counterattack_opportunity_reward"][rew_index] 

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
