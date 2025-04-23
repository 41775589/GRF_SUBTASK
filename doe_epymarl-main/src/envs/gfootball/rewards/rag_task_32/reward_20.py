import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense crossing and sprinting reward specific for wingers.
    It focuses on encouraging wingers to perform crossing accurately at high speeds.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Add the current state of reward wrapper to the pickle.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state from the pickle during loading.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on winger activity involving sprinting and crossing.
        Base reward components include specific actions such as dribbling (action 9)
        and moving right rapidly (actions 4 and 8 for sprint).
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward),
                      "cross_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]

            # Encourage sprinting down the wings
            if o['sticky_actions'][4] and o['sticky_actions'][8]:  # Right and sprint
                components["sprint_reward"][i] = 0.05
                reward[i] += components["sprint_reward"][i]

            # Cross completion from wings, with higher reward if closer to goal
            distance_to_goal = 1 - abs(o['ball'][0])
            if o['ball'][0] > 0.5 and o['game_mode'] == 4:  # Assuming right side cross
                components["cross_reward"][i] = 0.1 * distance_to_goal
                reward[i] += components["cross_reward"][i]

        return reward, components

    def step(self, action):
        """
        Execute a step using the given action, modify the reward, and return the results.
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
