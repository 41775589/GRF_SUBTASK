import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive actions with special focus on sliding and sprinting."""

    def __init__(self, env):
        super().__init__(env)
        self.sliding_coefficient = 2.0
        self.sprinting_coefficient = 1.5
        self.distance_threshold = 0.10
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the sticky actions counter and collected checkpoints when the environment is reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Pickle the current state along with the state of CheckpointRewardWrapper."""
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions": self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the internal state from a pickle."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle.get('CheckpointRewardWrapper', {}).get("sticky_actions", []), dtype=int)
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on defensive actions, particularly focusing on sliding and sprinting."""

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "sliding_reward": [0.0],
            "sprinting_reward": [0.0]
        }

        for obs in observation:
            # Check for sliding action
            if 'sticky_actions' in obs and obs['sticky_actions'][7] == 1:  # Assuming index 7 for sliding
                components["sliding_reward"][0] = self.sliding_coefficient

            # Check for sprinting action
            if 'sticky_actions' in obs and obs['sticky_actions'][8] == 1:  # Assuming index 8 for sprinting
                components["sprinting_reward"][0] = self.sprinting_coefficient
            
            # Calculate additional rewards for positioning closer to defensive objectives
            if 'left_team' in obs:
                own_player_pos = obs['left_team'][obs['active']]
                ball_position = obs['ball'][:2]
                distance_to_ball = np.linalg.norm(own_player_pos - ball_position)
                
                if distance_to_ball < self.distance_threshold:
                    components["sprinting_reward"][0] += self.sprinting_coefficient  # Encourage sprinting towards the ball
                    
        # Sum total reward considering components
        reward[0] = (components["base_score_reward"][0] + components["sliding_reward"][0] 
                     + components["sprinting_reward"][0])

        return reward, components

    def step(self, action):
        """Capture the action and its outcome, modify the reward accordingly, and return modified observations and info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update info with sticky actions count
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
