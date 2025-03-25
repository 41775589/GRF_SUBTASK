import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for aggressive offensive maneuvers with dynamic adaptation
    during various game phases.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previously_owned = False
  
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previously_owned = False
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_maneuver_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Iterate over the observations for each agent, which are assumed to be 2.
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage ball possession
            if o['ball_owned_team'] == o['active']:
                if not self.previously_owned:
                    components["offensive_maneuver_reward"][rew_index] = 0.5
                    self.previously_owned = True
            else:
                self.previously_owned = False

            # Encourage quick attack, specifically by fast transitions into the opponent's half
            if o['ball'][0] > 0 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
                components["offensive_maneuver_reward"][rew_index] += 0.2

            # General forward movement with the ball
            if o['ball_owned_team'] == o['active'] and o['ball'][0] > 0.5:
                components["offensive_maneuver_reward"][rew_index] += 0.1

            # Combine base score reward with the maneuver reward
            reward[rew_index] += components["offensive_maneuver_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'previously_owned': self.previously_owned}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previously_owned = from_pickle['CheckpointRewardWrapper']['previously_owned']
        return from_pickle
