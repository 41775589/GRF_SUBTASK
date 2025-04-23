import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on maintaining possession under pressure and exploiting open spaces."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_counter = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['pass_completion_counter'] = self.pass_completion_counter
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_completion_counter = from_pickle['pass_completion_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_reward": [0.0] * len(reward),
                      "open_space_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            team_possession = o['ball_owned_team']
            ball_owner = o['ball_owned_player']

            # Encouraging passing under pressure and receiving
            if team_possession == 1 and ball_owner == o['active']:
                if not self.pass_completion_counter.get(rew_index):
                    self.pass_completion_counter[rew_index] = 0
                self.pass_completion_counter[rew_index] += 1
                components['possession_reward'][rew_index] = 0.05 * self.pass_completion_counter[rew_index]

            # Reward for advancing into open space
            open_spaces = self.calculate_open_spaces(o['right_team'], o['ball'])
            components['open_space_reward'][rew_index] = 0.1 * open_spaces

            # Calculate final rewards.
            reward[rew_index] += components['possession_reward'][rew_index] + components['open_space_reward'][rew_index]

        return reward, components

    def calculate_open_spaces(self, team_positions, ball_position):
        """Dummy function calculating 'openess' of the player (might be based on distance to closest opponents etc.)"""
        distances = np.linalg.norm(team_positions - ball_position[:2], axis=1)
        return np.mean(distances)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        new_observation = self.env.unwrapped.observation()
        
        # Update sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in new_observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
