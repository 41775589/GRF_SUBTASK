import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards advanced ball control and effective passing under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        # Define checkpoints within the game focusing on passing skills
        self.passing_accuracy_thresholds = [0.2, 0.5, 0.8] # Different levels of accuracy in passing
        self.passing_reward = [0.2, 0.3, 0.5] # Corresponding rewards for each accuracy level
        self.ball_control_reward = 0.1  # Reward for good ball control under pressure
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.sticky_actions_counter = state['sticky_actions_counter']
        self.env.set_state(state)
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components[f"passing_reward_{rew_index}"] = 0.0
            components[f"ball_control_reward_{rew_index}"] = 0.0
            
            # High, long, and short passes need to be rewarded if successfully completed
            if o['game_mode'] in [3, 4, 6]:  # Assuming modes 3, 4, 6 correspond to passing scenarios
                # Distance check (hypothetical function: evaluate_passing_effectiveness)
                pass_accuracy = self.evaluate_passing_effectiveness(o)
                reward[rew_index] += self.calculate_passing_reward(pass_accuracy)
                components[f"passing_reward_{rew_index}"] = self.calculate_passing_reward(pass_accuracy)
            
            # Ball control under pressure evaluation
            if self.is_under_pressure(o):
                reward[rew_index] += self.ball_control_reward
                components[f"ball_control_reward_{rew_index}"] = self.ball_control_reward

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def evaluate_passing_effectiveness(self, observation):
        # Dummy function, returns a random accuracy representation
        return np.random.choice(self.passing_accuracy_thresholds, p=[0.3, 0.4, 0.3])

    def calculate_passing_reward(self, accuracy):
        # Determine the reward based on accuracy of pass
        for threshold, reward in zip(self.passing_accuracy_thresholds, self.passing_reward):
            if accuracy >= threshold:
                return reward
        return 0

    def is_under_pressure(self, observation):
        # Dummy check for whether the player with the ball is under pressure
        return np.random.choice([True, False], p=[0.5, 0.5])
