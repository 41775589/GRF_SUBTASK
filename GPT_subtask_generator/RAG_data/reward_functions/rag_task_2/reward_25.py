import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that increases the reinforcement for maintaining strategic defensive positions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.defensive_zones = 3
        # Rewards for being in the defensive zone, ceasing opponents' advance and successful tackles
        self.defensive_zone_reward = 0.3
        self.ball_intercept_reward = 0.5
        self.successful_tackle_reward = 0.7
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
                      "defensive_zone_reward": [0.0] * len(reward),
                      "ball_intercept_reward": [0.0] * len(reward),
                      "successful_tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']]

            # Defensive zone incentives
            if player_pos[0] < -0.7:  # Adjust this threshold for your need
                components['defensive_zone_reward'][rew_index] = self.defensive_zone_reward
                reward[rew_index] += self.defensive_zone_reward
            
            # Rewards for intercepting the ball in a defensive area
            if o['ball_owned_team'] == 1:  # ball owned by the opposing team
                ball_pos = o['ball']
                if np.linalg.norm(player_pos - ball_pos[:2]) < 0.1 and player_pos[0] < -0.7:
                    components['ball_intercept_reward'][rew_index] = self.ball_intercept_reward
                    reward[rew_index] += self.ball_intercept_reward

            # Rewards on successful tackle
            if 'tackle' in o['sticky_actions'] and o['sticky_actions']['tackle']:
                components['successful_tackle_reward'][rew_index] = self.successful_tackle_reward
                reward[rew_index] += self.successful_tackle_reward

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
