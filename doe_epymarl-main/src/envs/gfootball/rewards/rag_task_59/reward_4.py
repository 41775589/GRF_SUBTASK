import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward system to focus on goalkeeper coordination."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_backing_up_reward = 0.5
        self.clearing_reward = 1.0
        self.goalkeeper_positions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['GoalkeeperPositions'] = self.goalkeeper_positions
        to_pickle['StickyActionsCounter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_positions = from_pickle['GoalkeeperPositions']
        self.sticky_actions_counter = from_pickle['StickyActionsCounter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_backing_up_reward": [0.0] * len(reward),
            "clearing_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Identify goalkeeper and track movements
            goalkeeper_idx = next((i for i, role in enumerate(o['right_team_roles']) if role == 0), None)
            if goalkeeper_idx is not None:
                current_position = o['right_team'][goalkeeper_idx]
                previous_position = self.goalkeeper_positions.get(rew_index, current_position)
                self.goalkeeper_positions[rew_index] = current_position
                # Check if goalkeeper is backing up closer to goal
                if current_position[0] > previous_position[0]:
                    components["goalkeeper_backing_up_reward"][rew_index] = self.goalkeeper_backing_up_reward
                    reward[rew_index] += components["goalkeeper_backing_up_reward"][rew_index]
            
            # Reward for clearing balls under high pressure
            if o['game_mode'] in [2, 3, 4, 5]:  # Modes that might indicate high pressure
                if 'ball_owned_player' in o and o['ball_owned_player'] == goalkeeper_idx:
                    components["clearing_reward"][rew_index] = self.clearing_reward
                    reward[rew_index] += components["clearing_reward"][rew_index]

        return reward, components

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
