import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for the task of mastering stop-sprint and stop-moving techniques.
    It provides rewards for abrupt stopping in response to directional changes, simulating defensive maneuvers.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
        components = {"base_score_reward": reward.copy(),
                      "defensive_maneuver_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the active players just stopped or sprinted (actions 8 and 9 are for sprint and dribble respectively)
            if ('sticky_actions' in o and (o['sticky_actions'][8] == 1 or o['sticky_actions'][9] == 1)):
                # Increase sticky action counter if current actions are sprint or stop
                self.sticky_actions_counter += o['sticky_actions'][8:10]

            # Reward given for stopping right after a sprint suggesting an abrupt stop in defensive scenario
            if self.sticky_actions_counter[8] > 0 and o['sticky_actions'][8] == 0:
                defensive_component = 0.2  # Reward component for this defensive stop
                components["defensive_maneuver_reward"][rew_index] = defensive_component
                reward[rew_index] += defensive_component

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
