import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards focusing on effective short passing under defensive pressure,
    aiming for ball retention and successful completion of passes in pressure situations.
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialize variables to track passing success
        self.passes_completed = 0
        self.pass_attempts = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward_increment = 0.3  # Reward for completing a pass under pressure

    def reset(self):
        """Reset the environment and tracking variables."""
        self.passes_completed = 0
        self.pass_attempts = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of this reward wrapper."""
        to_pickle['passes_completed'] = self.passes_completed
        to_pickle['pass_attempts'] = self.pass_attempts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of this reward wrapper."""
        from_pickle = self.env.set_state(state)
        self.passes_completed = from_pickle['passes_completed']
        self.pass_attempts = from_pickle['pass_attempts']
        return from_pickle

    def reward(self, reward):
        """
        Adds additional rewards for completing a pass when under defensive pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Check if in possession and just released the ball
            if o['ball_owned_team'] == 0 and o['sticky_actions'][9] == 1:  # Assuming action 9 is pass action
                self.pass_attempts += 1
                next_observation = self.env.unwrapped.observation()
                
                # Check if the pass was successful by seeing if the team still owns ball after action processed
                if next_observation[rew_index]['ball_owned_team'] == 0:
                    self.passes_completed += 1
                    components["pass_completion_reward"][rew_index] = self.pass_reward_increment
                    reward[rew_index] += self.pass_reward_increment
                    
        return reward, components

    def step(self, action):
        """
        Steps through the environment, applies custom reward adjustments, and records action updates.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
