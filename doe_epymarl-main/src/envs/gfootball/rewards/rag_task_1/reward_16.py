import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on offensive maneuvers and dynamic adaptation 
    during varied game phases for highly active attacking strategies.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.train_effective_attacks = 0.05
        self.forward_progression_bonus = 0.02
        self.smart_play_bonus = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "train_effective_attacks": [0.0] * len(reward),
                      "forward_progression": [0.0] * len(reward),
                      "smart_play": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # Evaluate ball position in favor of forward attacks
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and o['ball'][0] > 0:
                distance_to_goal = abs(o['ball'][0] - 1)
                if distance_to_goal < 0.3:
                    components["train_effective_attacks"][rew_index] = self.train_effective_attacks

            # Encouraging passing and strategic moves
            if 'sticky_actions' in o and np.sum(o['sticky_actions'][8:10]) > 0:  # active sprint/dribble
                components["smart_play"][rew_index] = self.smart_play_bonus

            # Forward movement in attacking direction (down the field towards opposite goal)
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                components["forward_progression"][rew_index] = self.forward_progression_bonus

            reward[rew_index] += (components["train_effective_attacks"][rew_index] +
                                  components["forward_progression"][rew_index] +
                                  components["smart_play"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
