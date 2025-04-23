import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to specialize in enhancing agents' ability to regain ball possession through standing tackles,
    focusing on precision and control during both normal gameplay and set-piece defense scenarios.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_successful_reward = 1.0
        self.tackle_attempt_penalty = -0.1
        self.ball_steal_multiplier = 2.0

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
                      "tackle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # We focus on increasing reward when a tackle is successful
            action_taken = obs['sticky_actions'][9]  # Assuming index 9 is for tackle action
            ball_possession_change = obs['ball_owned_team'] == 0 and self.previous_ball_owned_team == 1
            
            if action_taken:
                if ball_possession_change:
                    # Successfully tackled and gained ball possession
                    components['tackle_reward'][rew_index] = (self.tackle_successful_reward
                                                            * self.ball_steal_multiplier)
                else:
                    # Tackle attempt without success
                    components['tackle_reward'][rew_index] = self.tackle_attempt_penalty
                
                reward[rew_index] += components['tackle_reward'][rew_index]
        
        self.previous_ball_owned_team = obs['ball_owned_team']
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
