import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for goalkeeper training tasks 
    including shot stopping, quick decision-making, and communication with defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Counters for key skill elements
        self.save_attempts = 0
        self.quick_decisions = 0
        self.communications = 0
        self._goalkeeper_position = None

        # Introducing coefficients for different rewards
        self.save_coeff = 1.0
        self.decision_coeff = 0.5
        self.communication_coeff = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.save_attempts = 0
        self.quick_decisions = 0
        self.communications = 0
        self._goalkeeper_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'save_attempts': self.save_attempts,
            'quick_decisions': self.quick_decisions,
            'communications': self.communications,
            'goalkeeper_position': self._goalkeeper_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.save_attempts = state_data['save_attempts']
        self.quick_decisions = state_data['quick_decisions']
        self.communications = state_data['communications']
        self._goalkeeper_position = state_data['goalkeeper_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": [reward] * 2,
            "save_reward": [0.0] * 2,
            "decision_reward": [0.0] * 2,
            "communication_reward": [0.0] * 2
        }

        if observation is None:
            return reward, components

        player_obs = observation[0]  # Assuming the first player is the goalkeeper
        if player_obs['ball_owned_team'] == 0 and player_obs['active'] == player_obs['ball_owned_player']:
            # Goalkeeper has the ball
            self.quick_decisions += 1
            components['decision_reward'][0] = self.decision_coeff

        if player_obs['game_mode'] == 1:  # If there's an event like a shot on goal
            self.save_attempts += 1
            components['save_reward'][0] = self.save_coeff

        # Communication evaluation could be simulated by checking positional adjustments
        if self._goalkeeper_position and np.linalg.norm(self._goalkeeper_position - player_obs['left_team'][0]) > 0.1:
            self.communications += 1
            components['communication_reward'][0] = self.communication_coeff

        self._goalkeeper_position = player_obs['left_team'][0]  # Update last known position of GK

        for key in components:
            reward += components[key][0]  # Sum components for agent 0, modify list for multi-agent

        return [reward] * 2, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
