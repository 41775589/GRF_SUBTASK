import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward targeting wingers for crossing and sprinting abilities."""
    
    def __init__(self, env):
        super().__init__(env)
        self.winger_threshold = 0.6  # X threshold to consider player a winger in position
        self.crossing_reward = 0.1
        self.sprint_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_state = self.env.set_state(state)
        return from_state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            winger_position = abs(o['left_team'][o['active']][0]) > self.winger_threshold
            
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and winger_position:
                # Crossing ability: Increased reward if crossing (pass action performed from wings)
                if 'action' in o and o['action'] in [5, 6]:  # 5 and 6 are assumed cross-like actions
                    components["crossing_reward"][rew_index] = self.crossing_reward
                    reward[rew_index] += components["crossing_reward"][rew_index]
                
                # Sprint ability: Assuming a winger uses sprint (action 8) along the wing
                if 'sticky_actions' in o and o['sticky_actions'][8] == 1:
                    components["sprint_reward"][rew_index] = self.sprint_reward
                    reward[rew_index] += components["sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
