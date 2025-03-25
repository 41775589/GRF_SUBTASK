import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining seamless transitions and controlled pace in central midfield."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.cent_midfielder_positions = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.cent_midfielder_positions = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'cent_midfielder_positions': self.cent_midfielder_positions,
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.cent_midfielder_positions = from_pickle['CheckpointRewardWrapper']['cent_midfielder_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'transition_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            controlled_player_pos = o['left_team'][o['active']]
            midfield_index = np.where(o['left_team_roles'] == 5)[0]  # CM - central midfield
            midfield_positions = o['left_team'][midfield_index]
            
            # Checking distances between players for seamless transitions
            if np.any(midfield_positions):
                distance_to_midfielders = np.linalg.norm(midfield_positions - controlled_player_pos, axis=1)
                close_to_midfielder = np.any(distance_to_midfielders < 0.1)
                
                if close_to_midfielder:
                    components['transition_reward'][rew_index] = 0.05
                    reward[rew_index] += components['transition_reward'][rew_index]

            # Tracking if controlled player maintains a controlled pace
            controlled_player_speed = np.linalg.norm(o['left_team_direction'][o['active']])
            if controlled_player_speed < 0.01:
                components['transition_reward'][rew_index] += 0.05
                reward[rew_index] += 0.05

            # Update historical positions to track effective midfield control
            self.cent_midfielder_positions.append(controlled_player_pos)

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
