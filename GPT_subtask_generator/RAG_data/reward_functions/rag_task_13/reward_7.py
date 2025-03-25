import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards focused on the stopper role, emphasizing blocking,
    intercepting, and reducing forward moves by the opposing team.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = to_pickle.get("state", {})
        state['CheckpointRewardWrapper'] = {}
        return self.env.get_state(state)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Unpack the observation from the environment
        observation = self.env.unwrapped.get_observation()

        # Initialize reward components
        components = {
            "base_score_reward": reward,  # Original reward
            "intervention_reward": 0.0,   # Reward for blocking or intercepting
            "positioning_reward": 0.0     # Reward for positioning relative to opponents and ball
        }

        # Calculate rewards related to the stopper role
        if observation:
            # Extract useful observations
            active_player_pos = observation['left_team'][observation['active']]
            ball_pos = observation['ball'][:2]  # Only x, y coordinates
            opponents = observation['right_team']

            # Check for interventions and good defensive position
            dist_to_ball = np.linalg.norm(active_player_pos - ball_pos)
            components['positioning_reward'] = -dist_to_ball * 0.1

            # Checking if close enough to intercept/block the ball
            if dist_to_ball < 0.03:
                components['intervention_reward'] = 1.0

            # Sum all components of the reward
            total_reward = sum(components.values())
        else:
            total_reward = reward

        return total_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
