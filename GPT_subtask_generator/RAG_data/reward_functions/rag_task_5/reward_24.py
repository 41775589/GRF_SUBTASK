import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for defensive actions and quick transitions 
    to counter-attacks.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards for intercepting the ball and successful counter-attacks.
        self._interception_reward = 0.3
        self._counter_attack_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "interception_reward": [0.0] * len(reward), "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage intercepting the ball.
            if o['ball_owned_team'] != o['active'] and o['designated'] != self.sticky_actions_counter[8]:
                components["interception_reward"][rew_index] = self._interception_reward
                reward[rew_index] += components["interception_reward"][rew_index]
            
            # Reward for quick transition to attack after gaining possession.
            if self.sticky_actions_counter[9] == 1:  # When dribble action is toggled to true.
                components["counter_attack_reward"][rew_index] = self._counter_attack_reward
                reward[rew_index] += components["counter_attack_reward"][rew_index]
        
        # Update sticky actions counters for next step.
        sticky_actions = [player_obs['sticky_actions'] for player_obs in observation]
        self.sticky_actions_counter = np.array(sticky_actions).sum(axis=0)
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
