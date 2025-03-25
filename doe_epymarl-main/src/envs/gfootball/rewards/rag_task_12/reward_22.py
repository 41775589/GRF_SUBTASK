import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom wrapper for training a midfield/advance defender agent with specific abilities."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward constants
        self.dribble_reward = 0.1
        self.high_pass_reward = 0.2
        self.long_pass_reward = 0.2
        self.sprint_reward = 0.05
        self.stop_sprint_reward = 0.05
        
    def reset(self):
        """Reset the wrapper's state and the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for serialization."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from deserialization."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Augment the reward based on specific agent actions and positions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation), "Reward and observation lengths must match."
        
        additional_rewards = [0.0] * len(reward)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Adjust reward based on specific actions performed
            if o['sticky_actions'][8]:  # Sprint action
                additional_rewards[rew_index] += self.sprint_reward
            if o['sticky_actions'][9]:  # Dribble under pressure
                additional_rewards[rew_index] += self.dribble_reward
            game_mode = o['game_mode']
            if game_mode in [2, 3]:  # High pass and long pass situations
                if game_mode == 2:
                    additional_rewards[rew_index] += self.high_pass_reward
                elif game_mode == 3:
                    additional_rewards[rew_index] += self.long_pass_reward
           
            reward[rew_index] += additional_rewards[rew_index]
        
        return reward, {"additional_rewards": additional_rewards}
    
    def step(self, action):
        """Step through the environment and augment reward at each step."""
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
