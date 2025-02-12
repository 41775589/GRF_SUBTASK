import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a complex reward for a soccer midfielder/defender agent focused on transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_accuracy_bonus = 0.1
        self.dribble_success_bonus = 0.05
        self.sprint_efficiency_bonus = 0.02

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_bonus": [0.0] * len(reward),
                      "dribble_success_bonus": [0.0] * len(reward),
                      "sprint_efficiency_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if passing actions are effective (ball possession and pass completion)
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                components["pass_accuracy_bonus"][rew_index] = self.pass_accuracy_bonus
                reward[rew_index] += components["pass_accuracy_bonus"][rew_index]
            
            # Check dribble effectiveness (maintaining ball possession under pressure)
            if 'sticky_actions' in o and np.any(o['sticky_actions'][1:3]):
                components["dribble_success_bonus"][rew_index] = self.dribble_success_bonus
                reward[rew_index] += components["dribble_success_bonus"][rew_index]
            
            # Check sprint efficiency (usage during specific game modes/positions)
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:
                components["sprint_efficiency_bonus"][rew_index] = self.sprint_efficiency_bonus
                reward[rew_index] += components["sprint_efficiency_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        info["final_reward"] = sum(modified_reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, modified_reward, done, info
