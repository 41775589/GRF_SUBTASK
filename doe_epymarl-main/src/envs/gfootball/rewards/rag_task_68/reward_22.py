import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds detailed rewards based on offensive strategies and ball control.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "shooting_efficiency": [0.0] * len(reward),
                      "dribble_efficiency": [0.0] * len(reward),
                      "passing_efficiency": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            game_mode = o.get('game_mode', 0)

            # Define increased reward for successful shots: mode = Penalty or Goal
            if game_mode in {6} and o['score'][1] > o['score'][0]:  # Initialized as right team scoring
                components["shooting_efficiency"][rew_index] = 1.0

            # Enhance reward for dribbling actions
            if o['sticky_actions'][9] == 1:  # Dribble action initiated
                components["dribble_efficiency"][rew_index] = 0.2 
            else:
                components["dribble_efficiency"][rew_index] = -0.1 

            # Pass efficiency: check changes in ball possession between left and right team
            if o['ball_owned_team'] == 1 and np.abs(o['ball'][0]) > 0.5:  # Ball is considerably forward in field
                if o['right_team'][o['ball_owned_player']][0] > 0.5:  # Player with ball is forward
                    previous_player_with_ball = self.last_ball_owned_player[rew_index] if hasattr(self, 'last_ball_owned_player') else None
                    if previous_player_with_ball is not None and previous_player_with_ball != o['ball_owned_player']:
                        components["passing_efficiency"][rew_index] = 0.3
            self.last_ball_owned_player = o['ball_owned_player']
            
            # Updating final reward by adding components
            reward[rew_index] += components["shooting_efficiency"][rew_index]
            reward[rew_index] += components["dribble_efficiency"][rew_index]
            reward[rew_index] += components["passing_efficiency"][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
