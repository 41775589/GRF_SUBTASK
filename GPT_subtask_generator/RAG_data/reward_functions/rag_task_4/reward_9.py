import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for training advanced dribbling and evasion skills."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._total_distance_reward = 0.0
        self._dribbling_reward = 0.1
        self._sprint_reward = 0.05
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle
    
    def reward(self, reward: list[float]) -> tuple[list[float], dict[str, list[float]]]:
        '''Adjusts rewards based on dribbling, evasion with sprint, and ball distance handling.'''
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward),
                      "evasion_distance_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            controlled_player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            opponent_players = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']

            # Calculate dribbling reward
            if o['sticky_actions'][9]:  # Index 9 refers to the dribbling action
                components["dribbling_reward"][rew_index] = self._dribbling_reward
                reward[rew_index] += components["dribbling_reward"][rew_index]
            
            # Calculate sprint reward
            if o['sticky_actions'][8]:  # Index 8 refers to the sprint action
                components["sprint_reward"][rew_index] = self._sprint_reward
                reward[rew_index] += components["sprint_reward"][rew_index]

            # Additional reward for evasion based on distance maintained or increased from opponents
            min_distance = min([np.linalg.norm(opponent_pos - controlled_player_pos) for opponent_pos in opponent_players])
            components["evasion_distance_reward"][rew_index] = min_distance * 0.1  # Scaled by distance
            reward[rew_index] += components["evasion_distance_reward"][rew_index]

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
            for i, action in enumerate(agent_obs["sticky_actions"]):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
