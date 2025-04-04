import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining controlled pace and seamless transitions in midfield."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pace_control_multiplier = 0.5
        self.transition_reward = 1.0
        
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

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pace_control_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for player_idx in range(len(reward)):
            o = observation[player_idx]

            # Reward for maintaining controlled pace
            control = o.get('sticky_actions', [])
            if control[8] > 0 or control[9] > 0:  # Check for sprint or dribble actions
                components["pace_control_reward"][player_idx] = self.pace_control_multiplier
                reward[player_idx] += components["pace_control_reward"][player_idx]
            
            # Reward for seamless transitions
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["transition_reward"][player_idx] = self.transition_reward
                reward[player_idx] += components["transition_reward"][player_idx]

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
