import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages offensive skills like Short Pass, Long Pass, 
    Shot, Dribble, and Sprint to create scoring opportunities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Coefficients for different actions
        self.pass_coefficient = 0.2
        self.shot_coefficient = 0.5
        self.dribble_coefficient = 0.1
        self.sprint_coefficient = 0.1
        # Initialize a tracker for sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset the sticky actions counter on environment reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the current state of this wrapper along with the environment state
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Load the state of this wrapper along with the environment state
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Apply the customized reward logic based on actions reported in 'sticky_actions'
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Check for different actions active for controlled players
            active_actions = o['sticky_actions']
            # Reward short and long pass actions
            if active_actions[0] or active_actions[1]:  # Assuming indices 0 and 1 are for short and long passes
                components["pass_reward"][rew_index] = self.pass_coefficient
                reward[rew_index] += components["pass_reward"][rew_index]
            # Reward shot action
            if active_actions[4]:  # Assuming index 4 is for the 'Shot' action
                components["shot_reward"][rew_index] += self.shot_coefficient
                reward[rew_index] += components["shot_reward"][rew_index]
            # Reward dribble action
            if active_actions[9]:  # Assuming index 9 is for the 'Dribble' action
                components["dribble_reward"][rew_index] += self.dribble_coefficient
                reward[rew_index] += components["dribble_reward"][rew_index]
            # Reward sprint action
            if active_actions[8]:  # Assuming index 8 is for the 'Sprint'
                components["sprint_reward"][rew_index] += self.sprint_coefficient
                reward[rew_index] += components["sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Get the observation, reward, done status, and info from the environment
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the custom logic
        reward, components = self.reward(reward)
        # Add final reward and components to the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset sticky actions counter
        self.sticky_actions_counter.fill(0)
        # Count currently active sticky actions for each agent
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
