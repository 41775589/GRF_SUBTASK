import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for executing high pass skills with precision.
    Both trajectory control and power assessment are considered for enhancing
    the agent's ability to perform high passes effectively in appropriate situations.
    """

    def __init__(self, env):
        super().__init__(env)
        # Custom attributes to evaluate pass quality
        self._high_pass_attempted = False
        self._successful_high_pass = False
        self.high_pass_coefficient = 0.2  # Weight of the high pass reward component
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For tracking the number of each type of sticky action

    def reset(self):
        """
        Resets the environment and any tracking variables.
        """
        self._high_pass_attempted = False
        self._successful_high_pass = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Returns the state along with any added components for saving/loading the game.
        """
        to_pickle['high_passes'] = (self._high_pass_attempted, self._successful_high_pass)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restores saved state along with any custom components relevant to this wrapper.
        """
        from_pickle = self.env.set_state(state)
        self._high_pass_attempted, self._successful_high_pass = from_pickle['high_passes']
        return from_pickle

    def reward(self, reward):
        """
        Modifies reward based on the agent's execution of high passes.
        """
        components = {"base_score_reward": reward.copy(),
                      "high_pass_skill_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            # Retrieve specific observation for each agent
            o = observation[rew_index]

            # Check if a high pass action is taken, using e.g., action attributes that are stateful
            if self.sticky_actions_counter[3] > 0:  # Assuming index 3 = action_high_pass (hypothetical)
                self._high_pass_attempted = True

            # Determine if the high pass was successful
            if self._high_pass_attempted:
                # We check if ball's z position indicates a high pass.
                ball_height = o['ball'][2]
                if ball_height > 0.1:  # Threshold for height to consider it a successful high pass
                    self._successful_high_pass = True
                    # Reward agent for successful execution based on conditions met
                    success_reward = self.high_pass_coefficient
                    reward[rew_index] += success_reward
                    components["high_pass_skill_reward"][rew_index] = success_reward

        return reward, components

    def step(self, action):
        """
        Step function that evaluates agents' actions and applies rewards.
        Appends reward components and tracking information to the info dictionary.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Aggregate final reward to info dictionary for external tracking/analysis
        info['final_reward'] = sum(reward)
        
        # Info fields for each component of the reward and details of the actions performed
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track sticky actions used by the agents
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
