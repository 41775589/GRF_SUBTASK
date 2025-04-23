import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on enhancing team defensive strategies primarily by training 
    the goalkeeper and defenders in a football environment.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.saves_made_by_goalkeeper = 0
        self.defensive_actions_by_defenders = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.saves_made_by_goalkeeper = 0
        self.defensive_actions_by_defenders = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['saves_made_by_goalkeeper'] = self.saves_made_by_goalkeeper
        state['defensive_actions_by_defenders'] = self.defensive_actions_by_defenders
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.saves_made_by_goalkeeper = from_pickle['saves_made_by_goalkeeper']
        self.defensive_actions_by_defenders = from_pickle['defensive_actions_by_defenders']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_save_reward": [0.0] * len(reward),
            "defender_action_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Ball is owned by the opponent (assuming left team is ours and controlled by agents)
                if o['right_team_roles'][o['active']] == 0:  # Role 0 is typically the goalkeeper
                    self.saves_made_by_goalkeeper += 1
                    components["goalkeeper_save_reward"][rew_index] = 1.0  # Reward for a goalkeeper making a save
                elif o['right_team_roles'][o['active']] in (1, 2, 3, 4):  # Defensive roles
                    self.defensive_actions_by_defenders += 1
                    components["defender_action_reward"][rew_index] = 0.5  # Reward for defenders taking defensive actions

            reward[rew_index] += components["goalkeeper_save_reward"][rew_index] + components["defender_action_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
