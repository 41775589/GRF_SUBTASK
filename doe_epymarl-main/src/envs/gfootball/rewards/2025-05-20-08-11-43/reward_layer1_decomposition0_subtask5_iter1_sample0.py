import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that incentivizes successful, accurate passes, particularly focusing on long passes and maintaining possession during transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int) 
        self.pass_success_reward = 0.5  # Reward for successful long passes
        self.pass_receipt_reward = 0.3   # Reward for receiving a pass
        self.previous_ball_owned_team = -1
        self.previous_ball_position = np.zeros(3)  # Keep track of the previous ball position

    def reset(self):
        """Resets the environment and clears the sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owned_team = -1
        self.previous_ball_position = np.zeros(3)
        return self.env.reset()

    def get_state(self, to_pickle):
        """State serialization for the environment."""
        to_pickle['previous_ball_owned_team'] = self.previous_ball_owned_team
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """State deserialization for the environment."""
        from_pickle = self.env.set_state(state)
        self.previous_ball_owned_team = from_pickle['previous_ball_owned_team']
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        """Custom reward function focusing on accurate passing and ball control."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_success_reward": [0.0],
            "pass_receipt_reward": [0.0]
        }

        if observation is None:
            return reward, components

        o = observation[0]
        ball_position = o['ball']

        # Reward for successfully sending a long pass
        if self.previous_ball_owned_team != o['ball_owned_team'] and o['ball_owned_team'] != -1:
            pass_distance = np.linalg.norm(ball_position - self.previous_ball_position)

            if pass_distance > 0.3:  # Assuming this threshold discerns long passes
                components["pass_success_reward"][0] = self.pass_success_reward

            # Check if the pass has been successfully received
            # This assumes we have a way to track pass reception accurately
            if o['ball_owned_team'] == self.previous_ball_owned_team:
                components["pass_receipt_reward"][0] = self.pass_receipt_reward  

        # Update state variables
        self.previous_ball_position = ball_position
        self.previous_ball_owned_team = o['ball_owned_team']

        # Calculate the total reward for the current step
        reward[0] += components["pass_success_reward"][0] + components["pass_receipt_reward"][0] 
        return reward, components

    def step(self, action):
        """Performs an action in the environment and processes the reward adjustments."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)

        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
