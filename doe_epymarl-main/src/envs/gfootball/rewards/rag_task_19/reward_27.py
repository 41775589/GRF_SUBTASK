import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive and midfield maneuvers."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize a counter for defensive and midfield actions
        self.defensive_actions_counter = 0
        self.midfield_actions_counter = 0
        self.defensive_reward_coefficient = 0.3
        self.midfield_reward_coefficient = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions_counter = 0
        self.midfield_actions_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state.update({
            'defensive_actions_counter': self.defensive_actions_counter,
            'midfield_actions_counter': self.midfield_actions_counter
        })
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_actions_counter = from_pickle['defensive_actions_counter']
        self.midfield_actions_counter = from_pickle['midfield_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": 0.0,
            "midfield_reward": 0.0
        }

        if observation is None:
            return reward, components

        ball_position = np.array(observation['ball'][:2])
        own_team_positions = observation['left_team']
        opponent_team_positions = observation['right_team']
        
        # Check ball possession to decide reward attribution (defensive or midfield)
        if observation['ball_owned_team'] == 0: # Own team has possession
            ball_handler_index = observation['ball_owned_player']
            ball_handler_position = own_team_positions[ball_handler_index]
            # Euclidean distance from own goal (normalized field coordinates)
            distance_from_own_goal = np.linalg.norm(ball_handler_position + np.array([1, 0]))

            # Reward defensive actions based on proximity to own goal 
            if distance_from_own_goal < 0.5:
                components["defensive_reward"] = self.defensive_reward_coefficient
                self.defensive_actions_counter += 1
            # Reward midfield control actions based on position on the field
            elif 0.5 <= distance_from_own_goal <= 0.75:
                components["midfield_reward"] = self.midfield_reward_coefficient
                self.midfield_actions_counter += 1

        # Add defensive and midfield rewards to the base reward
        reward += components["defensive_reward"]
        reward += components["midfield_reward"]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
