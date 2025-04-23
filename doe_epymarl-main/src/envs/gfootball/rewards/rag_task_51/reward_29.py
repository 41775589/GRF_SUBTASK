import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward function for goalkeeper training."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize any required variables here
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_index = 0  # Assuming the goalkeeper is the first player in the left_team lineup

    def reset(self):
        # Reset any counters and other variables
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save any state information required for this reward wrapper
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Load any state information needed for this reward wrapper
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """ Computes the modified reward, prioritizing goalkeeper performance."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_performance": [0.0, 0.0]}

        if observation is None:
            return reward, components
        
        # Only continue if the observation contains relevant data
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if the active player is the goalkeeper
            if o['active'] == self.goalkeeper_index:
                # Calculate reward for shot-stopping. Increasing the reward if the goalkeeper stops the goal when the ball is close.
                if (o['ball_owned_team'] == 1 and 
                        o['ball'][0] > 0.8 and
                        abs(o['ball'][1]) < 0.07):  # Ball near the goal and within the goal width
                    components["goalkeeper_performance"][rew_index] += 0.1
                # Reward for initiating counter-attacks with accurate passes
                if o['sticky_actions'][9] == 1:  # Assuming index 9 corresponds to passing/dribble
                    components["goalkeeper_performance"][rew_index] += 0.05

                # Modify the reward
                reward[rew_index] += components["goalkeeper_performance"][rew_index]

        return reward, components 

    def step(self, action):
        # Step through environment, adjust reward details accordingly
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
