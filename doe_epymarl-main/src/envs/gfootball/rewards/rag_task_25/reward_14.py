import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for dribbling techniques including using 
    the sprint action and maintaining control under pressure.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Keeps track of how many actions our agents have taken while sprinting
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Modifier for dribble reward
        self.dribble_reward_modifier = 0.1
        # Modifier for sprint reward
        self.sprint_reward_modifier = 0.1
        # Modifier for successful ball control when under pressure
        self.control_under_pressure_reward = 0.15

    def reset(self):
        # Reset the sprint actions counter and other necessary components
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modifies the reward based on agent performance in dribbling and sprinting scenarios."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward),
            "control_under_pressure_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Reward dribbling skill especially close to opponents
            if o['sticky_actions'][9] == 1:  # 9 = action dribble
                components["dribble_reward"][rew_index] = self.dribble_reward_modifier

            # Reward sprint usage, but only if it makes sense (team has ball, close to an opponent)
            if o['sticky_actions'][8] == 1 and o['ball_owned_team'] == 0:  # 8 = action sprint
                components["sprint_reward"][rew_index] = self.sprint_reward_modifier

            # Reward maintaining control under pressure (ball_control close to opposing players)
            if np.any((o['left_team'] - o['ball'][:2])**2 < 0.01):  # arbitrary threshold for "pressure"
                components["control_under_pressure_reward"][rew_index] = self.control_under_pressure_reward

            # Calculate total reward
            reward[rew_index] += (components["dribble_reward"][rew_index] + 
                                  components["sprint_reward"][rew_index] + 
                                  components["control_under_pressure_reward"][rew_index])

        return reward, components

    def step(self, action):
        # Take a step using the base environment
        observation, reward, done, info = self.env.step(action)
        # Recalculate the reward using the custom function
        reward, components = self.reward(reward)
        # Store the new reward and components in the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Track sticky actions counts
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
