import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for high passes and crosses."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Unlock reward when the ball is passed high or crossed effectively
        self.cross_pass_reward = 0.2

    def reset(self):
        # Reset sticky actions counter
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save any required state data
        base_state = self.env.get_state(to_pickle)
        base_state['sticky_actions_counter'] = self.sticky_actions_counter
        return base_state

    def set_state(self, state):
        # Load the saved state data
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on high passes and crossing behavior."""
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "cross_pass_reward": [0.0] * len(reward)}
        
        for i, o in enumerate(observation):
            # High passes - increase reward when the ball height and direction show a high trajectory
            if o['ball'][2] > 0.1 and np.linalg.norm(o['ball_direction']) > 0.1:
                components["cross_pass_reward"][i] = self.cross_pass_reward

            # Crosses - rewarding the ball reaching the other half while maintaining height and direction
            if o['ball'][0] > 0.7 and abs(o['ball'][1]) > 0.3 and o['ball'][2] > 0.1:
                components["cross_pass_reward"][i] += self.cross_pass_reward
            
            reward[i] += components["cross_pass_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        
        # Track sticky actions per active player
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
