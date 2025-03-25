import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long pass accuracy."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_reward = 0.1
        self.pass_distance_threshold = 0.5  # Encourages passes that are at least this fraction of field length

    def reset(self):
        """Resets the environment and clears the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Stores the current state of the environment in a pickle object."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state of the environment from a pickle object."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Calculates a new reward based on the accuracy and distance of passes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "pass_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if o['game_mode'] == 2 and o['ball_owned_team'] == 0:  # Only reward during the team's control (game modes may vary)
                ball_start = o['ball']
                # Successive environment's state needs to show the ball far away in the same possession event
                next_observation, _, _, _ = self.env.peek()  # Hypothetical method to peek next step without advancing env
                if next_observation:
                    ball_end = next_observation[i]['ball']
                    pass_distance = np.linalg.norm(ball_end[:2] - ball_start[:2])  # ignore z to simplify
                    if pass_distance > self.pass_distance_threshold:
                        components["pass_accuracy_reward"][i] = self.pass_accuracy_reward
                        reward[i] += components["pass_accuracy_reward"][i]

        return reward, components
    
    def step(self, action):
        """Executes one time step within the environment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
