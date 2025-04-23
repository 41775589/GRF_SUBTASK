import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for positioned play and possession changes."""

    def __init__(self, env):
        super().__init__(env)
        self.posession_change_counter = np.zeros(2, dtype=int)  # 0-index for left team, 1-index for right team
        self.position_checkpoints_collected = [{}, {}]  # Two dictionaries for each team
        self.checkpoint_value = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.posession_change_counter = np.zeros(2, dtype=int)
        self.position_checkpoints_collected = [{}, {}]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['posession_change_counter'] = self.posession_change_counter.tolist()
        to_pickle['position_checkpoints_collected'] = self.position_checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.posession_change_counter = np.array(from_pickle['posession_change_counter'], dtype=int)
        self.position_checkpoints_collected = from_pickle['position_checkpoints_collected']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Initialize components
        components = {"base_score_reward": reward.copy(),
                      "possession_change": [0.0] * 2,
                      "position_checkpoint": [0.0] * 2}
        if observation is None:
            return reward, components

        ball_owned_team = observation['ball_owned_team']
        if ball_owned_team in [0, 1]:  # If some team owns the ball
            if self.posession_change_counter[ball_owned_team] == 0:
                self.posession_change_counter[ball_owned_team] = 1
                self.posession_change_counter[1 - ball_owned_team] = 0
                components["possession_change"][ball_owned_team] = 0.1
                reward[ball_owned_team] += components["possession_change"][ball_owned_team]
        
        # Check the positions 
        for team_index, team in enumerate(['left_team', 'right_team']):
            if f'{team}_roles' in observation:
                for player_index, role in enumerate(observation[f'{team}_roles']):
                    if role in [1, 2, 3, 4]:
                        # Defenders gain positional rewards
                        key = f'{team_index}-{player_index}'
                        if key not in self.position_checkpoints_collected[team_index]:
                            components["position_checkpoint"][team_index] += self.checkpoint_value
                            reward[team_index] += components["position_checkpoint"][team_index]
                            self.position_checkpoints_collected[team_index][key] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action == 1:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
