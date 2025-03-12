import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that adds checkpoints to encourage offensive plays such as accurate shooting, effective dribbling, and diverse passing."""
    
    def __init__(self, env):
        super().__init__(env)
        self._checkpoint_reward = 0.1
        self._num_checkpoints = 5
        self.dribble_reward_increment = 0.05
        self.pass_reward_increment = 0.2
        self.shoot_reward_increment = 0.3
        
        # Reset collects checkpoints per episode
        self.collected_checkpoints = {}
    
    def reset(self):
        """Reset for a new episode."""
        self.collected_checkpoints = {}
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Get current state with collected checkpoints for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.collected_checkpoints.copy()
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Set state for the environment with checkpoints."""
        from_pickle = self.env.set_state(state)
        self.collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        """Calculate and augment the reward based on offensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
            "shoot_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        # Element-wise process rewards for each observation/agent
        for i, (obs, rw) in enumerate(zip(observation, reward)):
            if 'sticky_actions' in obs:
                # Dribbling logic: increase for dribble action
                if obs['sticky_actions'][9] == 1:  # 'action_dribble' is active
                    components["dribble_reward"][i] = self.dribble_reward_increment
                    
                # Passing logic: reward for long and high passes
                if obs['game_mode'] in [2, 3]:  # 'GameMode_GoalKick' or 'GameMode_FreeKick'
                    components["pass_reward"][i] = self.pass_reward_increment
                
                # Shoot logic: reward for shooting towards goal
                if obs['game_mode'] == 6:  # 'GameMode_Penalty'
                    components["shoot_reward"][i] = self.shoot_reward_increment
            
            # Applying calculated components to reward
            rw += components["dribble_reward"][i] + components["pass_reward"][i] + components["shoot_reward"][i]
            reward[i] = rw
        
        return reward, components

    def step(self, action):
        """Execute a step using the given action, then modify the reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Logging each reward component in 'info'
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
