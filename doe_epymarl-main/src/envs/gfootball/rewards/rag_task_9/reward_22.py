import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward based on football offense actions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Count for sticky actions which have non-zero indexes in football environment
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Initialize components dictionary to include in the output for better understanding reward parts
        components = {"base_score_reward": reward.copy(), "offensive_action_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for controlling the ball by the agent's team
            ball_control = 1 if o['ball_owned_team'] == 1 else 0
            
            # Rewards for specific actions related to passing, dribbling, shooting
            action_rewards = {
                7: 0.02,  # Bottom Left (Attempt of Dribble)
                8: 0.02,  # Dribbling
                9: 0.05,  # Sprinting
                6: 0.1,   # Shot
                3: 0.1    # Long Pass
            }
            
            for action, action_reward in action_rewards.items():
                if o['sticky_actions'][action] == 1:
                    components["offensive_action_reward"][rew_index] += action_reward
            
            # Combining rewards: base reward + action rewards + ball control reward
            reward[rew_index] = 0.5 * reward[rew_index] + 0.5 * ball_control + components["offensive_action_reward"][rew_index]

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
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
