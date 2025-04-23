import gym
import numpy as np
class DribblingSkillRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances dribbling skills against the goalkeeper. It incentivizes
    actions like dribbling, sudden changes in direction, and maintaining ball control.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Store the environment state.
        """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the environment's state.
        """
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Modify the reward function to emphasize dribbling skills and effective maneuvers.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # If the agent scores, there's a baseline reward.
            if reward[rew_index] == 1:
                reward[rew_index] += 1
                continue

            # Reward for maintaining ball control near opponent's goal area
            if o['ball_owned_team'] == 1 and abs(o['ball'][0]) > 0.5:
                # Additionally rewards dribbling when close to opponent's goalkeeper
                dribbling_action = o['sticky_actions'][9]  # Example index for dribble action
                if dribbling_action == 1:
                    components["dribbling_reward"][rew_index] = 0.2
                    reward[rew_index] += components["dribbling_reward"][rew_index]

            # Encourage sudden direction changes when close to opponent's goalkeeper
            distance_to_goal = np.linalg.norm(np.array([1, 0]) - o['ball'][:2])
            if distance_to_goal < 0.2:
                direction_change_reward = np.linalg.norm(o['ball_direction'][:2])
                if direction_change_reward > 0.1:  # Assuming some threshold for "sudden" change
                    components["dribbling_reward"][rew_index] += 0.5
                    reward[rew_index] += components["dribbling_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Execute a step in the environment, modify the reward, and append extra information.
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
