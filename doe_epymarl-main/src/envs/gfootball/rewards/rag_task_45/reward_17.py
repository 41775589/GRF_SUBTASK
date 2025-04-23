import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on defensive play with focus on sudden stops and sprints.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_for_stop = 0.1
        self.reward_for_sprint = 0.02
        self.last_action = None

    def reset(self):
        """
        Reset the environment and initialize the sticky action counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_action = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the wrapper along with other necessary states in pickle format.
        """
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter,
                                                "last_action": self.last_action}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the wrapper from the pickle data.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.last_action = from_pickle['CheckpointRewardWrapper']['last_action']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on the sudden stop and sprint actions executed.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_actions = o['sticky_actions']

            # Reward for stopping (action index for stop is 6)
            if self.last_action == 6 and active_actions[6] == 0:
                components["stop_reward"][rew_index] = self.reward_for_stop
                reward[rew_index] += components["stop_reward"][rew_index]

            # Reward for sprinting (action index for sprint is 8)
            if active_actions[8] == 1:
                components["sprint_reward"][rew_index] = self.reward_for_sprint
                reward[rew_index] += components["sprint_reward"][rew_index]

            self.last_action = np.argmax(active_actions) if np.any(active_actions) else None

        return reward, components

    def step(self, action):
        """
        Override the step function to include detailed breakdown of reward components.
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
