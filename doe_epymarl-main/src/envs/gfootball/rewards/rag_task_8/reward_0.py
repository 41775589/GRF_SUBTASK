import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the rewards for excelling in quick decision-making and efficient ball handling 
    to initiate counter-attacks immediately after recovering possession.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._recover_ball_reward = 1.0
        self._quick_pass_reward = 0.5
        self._previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_owner = [-1, -1]  # Reset ball ownership history
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_ball_owner': self._previous_ball_owner
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        data = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = data['sticky_actions_counter']
        self._previous_ball_owner = data['previous_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "recover_ball_reward": [0.0] * len(reward),
            "quick_pass_reward": [0.0] * len(reward)
        }

        if not observation:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Reward for recovering ball from opponent
            if o['ball_owned_team'] == 0 and self._previous_ball_owner[i] == 1:
                components["recover_ball_reward"][i] = self._recover_ball_reward
                reward[i] += components["recover_ball_reward"][i]
                
            # Reward for quick passing after ball recovery
            if self._previous_ball_owner[i] == 0 and o['game_mode'] in [3, 4, 5, 6]:  # Assumes game_modes for passing situations
                components["quick_pass_reward"][i] = self._quick_pass_reward
                reward[i] += components["quick_pass_reward"][i]

            # Update the previous ball owner states
            self._previous_ball_owner[i] = o['ball_owned_team']

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
