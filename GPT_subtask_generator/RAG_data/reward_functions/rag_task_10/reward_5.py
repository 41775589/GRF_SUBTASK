import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._interception_reward = 0.5
        self._stop_dribble_reward = 0.3
        self._marking_reward = 0.2
        self._tackle_reward = 0.4

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['StickyActionsCounter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['StickyActionsCounter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "stop_dribble_reward": [0.0] * len(reward),
                      "marking_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Assuming there are exactly two agents in the game
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and 'active' in o:
                # Reward for intercepting the ball
                if o['ball_owned_team'] != 0 and o['active'] == o['ball_owned_player']:
                    components["interception_reward"][rew_index] = self._interception_reward
                    reward[rew_index] += components["interception_reward"][rew_index]
                
                # Reward for marking opponent players closely
                opponent_team = 'right_team' if o['ball_owned_team'] == 0 else 'left_team'
                own_team = 'left_team' if o['ball_owned_team'] == 0 else 'right_team'
                for opp_player_pos in o[opponent_team]:
                    if np.linalg.norm(opp_player_pos - o[own_team][o['active']]) < 0.1:
                        components["marking_reward"][rew_index] = self._marking_reward
                        reward[rew_index] += components["marking_reward"][rew_index]
            
                # Reward for tackling (assuming tackle is a specific action)
                if o['sticky_actions'][8] == 1:  # Assuming action index '8' corresponds to tackling
                    components["tackle_reward"][rew_index] = self._tackle_reward
                    reward[rew_index] += components["tackle_reward"][rew_index]

                # Reward for stopping dribble specific scenarios
                if o['sticky_actions'][9] == 1 and o['active'] == o['ball_owned_player']:  # Assuming action index '9' is stopping dribble
                    components["stop_dribble_reward"][rew_index] = self._stop_dribble_reward
                    reward[rew_index] += components["stop_dribble_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
