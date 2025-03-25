import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for dribbling and sprinting effectively through defensive lines."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_distance_advanced = 0
        self.initial_player_pos = None
        self.dribble_reward_multiplier = 0.1
        self.sprint_reward_multiplier = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_distance_advanced = 0
        self.initial_player_pos = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist(),
            'total_distance_advanced': self.total_distance_advanced,
            'initial_player_pos': self.initial_player_pos
        }
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = np.array(wrapper_state['sticky_actions_counter'], dtype=int)
        self.total_distance_advanced = wrapper_state['total_distance_advanced']
        self.initial_player_pos = wrapper_state['initial_player_pos']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        current_player_pos = observation[0]['left_team'][observation[0]['active']]
        
        if self.initial_player_pos is None:
            self.initial_player_pos = current_player_pos
            
        distance_advanced = np.linalg.norm(self.initial_player_pos - current_player_pos)

        components = {'base_score_reward': reward.copy(),
                      'dribble_reward': [0.0] * len(reward),
                      'sprint_reward': [0.0] * len(reward)}

        sticky_actions = observation[0]['sticky_actions']
        is_dribbling = sticky_actions[9]
        is_sprinting = sticky_actions[8]

        if is_dribbling:
            components['dribble_reward'][0] = distance_advanced * self.dribble_reward_multiplier
        if is_sprinting:
            components['sprint_reward'][0] = distance_advanced * self.sprint_reward_multiplier

        reward[0] += components['dribble_reward'][0] + components['sprint_reward'][0]
        self.initial_player_pos = current_player_pos  # Update the position for the next calculation

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
