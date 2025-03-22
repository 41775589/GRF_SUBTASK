import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that promotes mastering offensive strategies including shooting accuracy, effective dribbling,
    and advanced passing strategies.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize parameters for rewarding passing, shooting, and dribbling based on game mechanics.
        self.pass_success_reward = 0.3
        self.shot_on_target_reward = 0.5
        self.effective_dribble_reward = 0.4
        self.score_reward = 1.0
    
    def reset(self):
        """
        Reset the environment and memory on call.
        """
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the environment for serialization if needed.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the environment from serialized data.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Rewriting the reward function to include dribbling, shooting, and passing rewards.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_success_reward": [0.0] * len(reward),
                      "shot_on_target_reward": [0.0] * len(reward),
                      "effective_dribble_reward": [0.0] * len(reward)}
        for idx, obs in enumerate(observation):
            # Check for effective shot
            if obs['game_mode'] in {6, 12}:  # Modes associated with shooting
                components['shot_on_target_reward'][idx] += self.shot_on_target_reward

            # Check for successful passes, modifies pass_success_reward
            # Note: The actual game observation does not detail every pass, this is a hypothetical implementation.
            if obs['ball_owned_team'] == obs['active']:  # Simple way to check if the player successfully controls the ball.
                components['pass_success_reward'][idx] += self.pass_success_reward

            # Reward for dribbling scenarios (also hypothetical logic)
            if obs.get('sticky_actions', [0]*10)[9] == 1:  # if dribbling action is active
                components['effective_dribble_reward'][idx] += self.effective_dribble_reward
            
            # Score updates
            if obs['score'][0] > 0 or obs['score'][1] > 0:
                reward[idx] += self.score_reward
        
        # Assign calculation back to reward variable along with modifications
        combined_rewards = np.array(reward) + np.array(components['pass_success_reward']) + \
                           np.array(components['shot_on_target_reward']) + \
                           np.array(components['effective_dribble_reward'])

        return list(combined_rewards), components

    def step(self, action):
        """
        Steps through the environment, altering the reward with custom reward enhancements.
        """
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the complex function logic defined above
        reward, components = self.reward(reward)
        
        # Store final combined reward calculations into info for proper tracking
        info['final_reward'] = sum(reward)
        # Add detailed components to the info
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        return observation, reward, done, info
