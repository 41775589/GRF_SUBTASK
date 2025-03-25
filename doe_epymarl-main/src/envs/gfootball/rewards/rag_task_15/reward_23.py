import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering long passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define the number of checkpoints and reward for each checkpoint
        self.checkpoint_rewards = np.linspace(0.1, 1.0, 10)
        self.positions_collected = {}

    def reset(self):
        """Reset the state of the environment and clear collected positions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positions_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Add wrapper state to the pickleable state of the environment."""
        to_pickle['CheckpointRewardWrapper'] = self.positions_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the environment from a pickle."""
        from_pickle = self.env.set_state(state)
        self.positions_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Compute the additional reward for the task of long passes."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_precision_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate the distance traveled by the ball
            ball_pos_start = np.array(o.get('ball', [0, 0, 0]))
            ball_pos_end = np.array(self.env.unwrapped.observation()[rew_index].get('ball', [0, 0, 0]))
            travel_distance = np.linalg.norm(ball_pos_end - ball_pos_start)

            # Check for successful long pass defined as travel over a certain threshold
            if travel_distance > 0.5:
                checkpoints = min(int(travel_distance * 10), len(self.checkpoint_rewards))
                if checkpoints not in self.positions_collected:
                    additional_reward = sum(self.checkpoint_rewards[:checkpoints])
                    components["long_pass_precision_reward"][rew_index] += additional_reward
                    self.positions_collected[checkpoints] = True

        # Update the actual reward
        adjusted_reward = [
            components["base_score_reward"][i] + components["long_pass_precision_reward"][i]
            for i in range(len(reward))
        ]
        return adjusted_reward, components

    def step(self, action):
        """Execute one time step within the environment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
