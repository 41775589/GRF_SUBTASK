import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for accurately executing long passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define parameters for rewarding long passes
        self.pass_distance_threshold = 0.5  # the pass should cover at least 50% of the field
        self.pass_accuracy_reward = 0.5      # reward for achieving the passing threshold
    
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
        # Preparing the output structure 
        base_score_rewards = reward.copy()
        additional_rewards = [0.0] * len(reward)
        
        components = {"base_score_reward": base_score_rewards,
                      "long_pass_accuracy_reward": additional_rewards}
        
        # Retrieve current observation from environment
        observations = self.env.unwrapped.observation()
        if observations is None:
            return reward, components
        
        for idx, o in enumerate(observations):
            # Only proceed if the ball was owned and passed by the active player
            if o['ball_owned_team'] != -1 and o['ball_owned_player'] == o['active']:
                # Analyze previous state to determine a change in possession
                # Here is a simplification since detailed tracking between frames is needed
                prev_ball_pos = o['ball']
                curr_ball_pos = self.env.unwrapped.observation()[idx]['ball']
                
                # Calculate the distance the ball traveled
                pass_distance = np.linalg.norm(
                    np.array(prev_ball_pos[:2]) - np.array(curr_ball_pos[:2])
                )
                
                # Reward if the pass crossed the distance threshold effectively
                if pass_distance > self.pass_distance_threshold:
                    additional_rewards[idx] += self.pass_accuracy_reward
                    reward[idx] += additional_rewards[idx]
        
        components = {
            "base_score_reward": base_score_rewards,
            "long_pass_accuracy_reward": additional_rewards
        }
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
