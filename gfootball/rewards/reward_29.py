import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward function for offensive football strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        # Rewards for various aspects of the offensive play
        self.shoot_reward = 1.0
        self.pass_reward = 0.5
        self.dribble_reward = 0.2
        
    def reset(self):
        """Reset environment and any internal state."""
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Serialize the internal state to support environment replication."""
        to_pickle['CheckpointRewardWrapper'] = {
            'shoot_reward': self.shoot_reward,
            'pass_reward': self.pass_reward,
            'dribble_reward': self.dribble_reward
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the internal state to support environment replication."""
        from_pickle = self.env.set_state(state)
        self.shoot_reward = from_pickle['CheckpointRewardWrapper']['shoot_reward']
        self.pass_reward = from_pickle['CheckpointRewardWrapper']['pass_reward']
        self.dribble_reward = from_pickle['CheckpointRewardWrapper']['dribble_reward']
        return from_pickle

    def reward(self, reward):
        """Modify the rewards based on the gameplay focused on offensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": np.array(reward),
            "shoot_reward": np.zeros_like(reward),
            "pass_reward": np.zeros_like(reward),
            "dribble_reward": np.zeros_like(reward)
        }
        if observation is None:
            return reward, components
        
        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0:  # When left team has the ball
                if obs['game_mode'] in {2, 3, 4}:  # Shoot situations (goal kick, free kick, corner)
                    reward[i] += self.shoot_reward
                    components["shoot_reward"][i] = self.shoot_reward
                if obs['sticky_actions'][6] or obs['sticky_actions'][7]:  # Dribble Actions
                    reward[i] += self.dribble_reward
                    components["dribble_reward"][i] = self.dribble_reward
                if obs['sticky_actions'][0] or obs['sticky_actions'][4]:  # Pass Actions (left or right pass)
                    reward[i] += self.pass_reward
                    components["pass_reward"][i] = self.pass_reward
        
        return reward, components
    
    def step(self, action):
        """Execute one time step within the environment."""
        observation, reward, done, info = self.env.step(action)
        new_reward, reward_components = self.reward(reward)
        info.update({
            'final_reward': sum(new_reward),
            'reward_components': reward_components
        })
        return observation, new_reward, done, info
