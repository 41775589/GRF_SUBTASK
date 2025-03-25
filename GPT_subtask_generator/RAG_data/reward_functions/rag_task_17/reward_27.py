import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for high-pass actions, maintaining wide positioning,
    and stretching the opposing defense. This is targeted at mastering wide midfield responsibilities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Retrieves the state of the wrapper along with the environment's state.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Sets the state of the environment and wrapper based on provided state.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Rewards the agents for effective wide midfield play by encouraging high passes
        and maintaining appropriate positioning to stretch the opposing defense.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, copy=True),
                      "wide_positioning": np.array([0.0, 0.0]),
                      "high_pass_bonus": np.array([0.0, 0.0])}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward wide positioning, particularly near sideline areas
            if np.abs(o['left_team'][o['active']][1]) > 0.3:
                components["wide_positioning"][i] = 0.05

            # Reward successful high passes
            if 'action_high_pass' in o['sticky_actions'] and o['sticky_actions']['action_high_pass'] == 1:
                components["high_pass_bonus"][i] = 0.1

            # Aggregate the rewards
            reward[i] += components["wide_positioning"][i] + components["high_pass_bonus"][i]

        return reward, components

    def step(self, action):
        """
        Steps through the environment, applies rewards, and passes modified rewards alongside observations.
        """
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
