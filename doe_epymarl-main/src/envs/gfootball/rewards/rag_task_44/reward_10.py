import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that halts AI movement precisely using Stop-Dribble as a defensive tactic to manage ball control under heavy pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Sticky actions are movements/actions that can persist over multiple frames

    def reset(self):
        """Reset initialized arrays on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Allows state serialization to be restored later."""
        to_pickle['CheckpointActionsCount'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore serialized state for precise control restoration."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointActionsCount']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on strict control of Stop-Dribble manoeuvres."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_dribble_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward agents for activation of Stop-Dribble under pressure
            if o['sticky_actions'][9] == 1 and self.sticky_actions_counter[9] < 3:  # Check if dribble is active and limit reward frequency
                components["stop_dribble_control_reward"][rew_index] = 0.05  # Incremental reward
                reward[rew_index] += components["stop_dribble_control_reward"][rew_index]
                self.sticky_actions_counter[9] += 1
            
            # Reset counter when dribble is not active
            if o['sticky_actions'][9] == 0:
                self.sticky_actions_counter[9] = 0
        
        return reward, components

    def step(self, action):
        """Overrides the 'step' function to incorporate our custom reward handling."""
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
