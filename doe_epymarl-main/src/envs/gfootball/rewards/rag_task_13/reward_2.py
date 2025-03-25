import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards the agent for effective defensive actions, focusing on the role of a 'stopper'.
    Rewards are based on effective man-marking, blocking shots, and stalling forward moves by opposing players.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocking_reward = 0.5
        self.marking_reward = 0.3
        self.stalling_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'StickyActions': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['StickyActions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "blocking_reward": [0.0] * len(reward),
                      "marking_reward": [0.0] * len(reward), "stalling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active'] and o['ball_owned_team'] != -1:
                ball_position = o['ball']
                player_position = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
                opp_player_positions = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']

                # Calculate distances to all opponents
                distances = np.linalg.norm(opp_player_positions - player_position, axis=1)

                # Check if marking closely
                if np.any(distances < 0.05):
                    components["marking_reward"][rew_index] = self.marking_reward
                    reward[rew_index] += components["marking_reward"][rew_index]

                # Reward for blocking potential shots
                if np.any(distances < 0.1) and o['ball'][0] > 0.9:
                    components["blocking_reward"][rew_index] = self.blocking_reward
                    reward[rew_index] += components["blocking_reward"][rew_index]
                
                # Reward for stalling opposition's forward movement
                if np.all(distances > 0.2) and o['ball'][0] < 0.5: 
                    components["stalling_reward"][rew_index] = self.stalling_reward
                    reward[rew_index] += components["stalling_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
