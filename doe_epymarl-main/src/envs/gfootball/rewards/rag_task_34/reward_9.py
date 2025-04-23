import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for shot precision and dribbling effectiveness against goalkeepers."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_precision_reward = 0.5  # Reward for shooting close to the goal center
        self.dribble_effectiveness_reward = 0.3  # Reward for successful dribbling near opponent's goalkeeper

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the state of the wrapper in the pickled state."""
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Retrieve and set the state of the wrapper from the pickled state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Customize reward function for shot precision and dribbling effectiveness."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": [r for r in reward],
                      "shot_precision_reward": [0.0] * len(reward),
                      "dribble_effectiveness_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Shot precision near the goal
            if o['game_mode'] == 6 and np.abs(o['ball'][1]) <= 0.044:  # Close to the center of the goal
                components["shot_precision_reward"][rew_index] += self.shot_precision_reward
                reward[rew_index] += components["shot_precision_reward"][rew_index]
            
            # Dribbling effectiveness
            if o['ball_owned_player'] == o['active'] and np.linalg.norm(o['ball'][:2]) > 0.9:
                components["dribble_effectiveness_reward"][rew_index] += self.dribble_effectiveness_reward
                reward[rew_index] += components["dribble_effectiveness_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Executes the environment's step function and applies the reward wrapper."""
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
