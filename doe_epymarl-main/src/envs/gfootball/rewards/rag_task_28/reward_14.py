import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focusing on dribbling skills in face-to-face situations with the goalkeeper.
    The reward emphasizes quick feints, sudden direction changes, and maintaining ball control under pressure.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize elements to track dribbling effectiveness and control under pressure.
        self.ball_control_rewards = 0.05
        self.feint_reward = 0.1

    def reset(self):
        # Reset sticky actions count on environment reset.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save wrapper-specific state.
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore wrapper-specific state.
        from_pickle = self.env.set_state(state)
        # Restore additional elements if needed
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function that evaluates dribbling and feinting performance.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward),
                      "feint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Applies enhanced rewards based on the dribbling and feints.
        for rew_index, o in enumerate(observation):
            if not o['ball_owned_team'] == (o['right_team'] if o['active'] in o['right_team'] else o['left_team']):
                continue
            
            # Check how good the dribbling was by evaluating the ball control and direction changes with sprints.
            if o['sticky_actions'][9]:  # Checking if dribble action is active.
                components["dribbling_reward"][rew_index] = self.ball_control_rewards
                reward[rew_index] += components["dribbling_reward"][rew_index]
            
            # Reward for feints: Change of direction while sprinting.
            if any(o['sticky_actions'][3:7]) and o['sticky_actions'][8]:  # Movement directions with sprint.
                components["feint_reward"][rew_index] = self.feint_reward
                reward[rew_index] += components["feint_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Process action as normal in the environment.
        observation, reward, done, info = self.env.step(action)
        enhanced_reward, components = self.reward(reward)
        
        # Add reward and components info into the info dictionary.
        info["final_reward"] = sum(enhanced_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Return outputs as per the gym environment requirements.
        return observation, enhanced_reward, done, info
