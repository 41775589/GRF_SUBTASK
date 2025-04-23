import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards advanced dribbling skills in direct confrontations with the goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_counter = 0
        self.dribble_reward = 0.2
        self.faint_reward = 0.15
        self.long_hold_penalty = -0.1
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_control_counter'] = self.ball_control_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_counter = from_pickle['ball_control_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "faint_reward": [0.0] * len(reward),
                      "long_hold_penalty": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Base reward as general game score
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # Checking if our player has the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['designated']:
                # Ball control reward
                self.ball_control_counter += 1
                
                if self.ball_control_counter > 50:
                    components["long_hold_penalty"][rew_index] = self.long_hold_penalty
                    reward[rew_index] += components["long_hold_penalty"][rew_index]
                elif self.ball_control_counter > 25:
                    # Reward dribbling if holding the ball for a significant time without losing it.
                    components["dribble_reward"][rew_index] = self.dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]
                
            # Reward for directional changes
            if np.sum(o['sticky_actions'][1:5]) > 1:  # more than one directional action taken
                components["faint_reward"][rew_index] = self.faint_reward
                reward[rew_index] += components["faint_reward"][rew_index]

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
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
