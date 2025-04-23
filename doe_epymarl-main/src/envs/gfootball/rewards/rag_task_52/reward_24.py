import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on enhancing defensive
    capabilities like tackling proficiency, movement control, and pressured passing tactics."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward scaling factors for specific defensive actions
        self.tackle_reward = 0.2  # Reward for successful tackles
        self.positioning_reward = 0.1 # Reward for being in a good defensive position
        self.pass_interception_reward = 0.3  # Reward for intercepting a pass
    
    def reset(self):
        """Reset environment and reward-related counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return the state with the wrapper's variables inclusive."""
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': list(self.sticky_actions_counter)}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from the provided state data with any necessary extractions for the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions'])
        return from_pickle
 
    def reward(self, reward):
        """Calculate and distribute dense defensive-related rewards."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward),
                      "pass_interception_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful tackles (hypothetical check on 'tackles_successful')
            if getattr(o, "tackles_successful", False):
                components["tackle_reward"][rew_index] += self.tackle_reward
            
            # Reward based on the player's positioning relative to the ball and goal area
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0:  # Simplified condition for ball in opponent half
                components["positioning_reward"][rew_index] += self.positioning_reward
            
            # Reward for intercepting a pass (hypothetical check on 'interceptions')
            if getattr(o, "interceptions", False):
                components["pass_interception_reward"][rew_index] += self.pass_interception_reward
            
            # Combine all rewards
            total_additional_reward = (1.0 * components["tackle_reward"][rew_index] +
                                       1.0 * components["positioning_reward"][rew_index] +
                                       1.5 * components["pass_interception_reward"][rew_index])

            reward[rew_index] += total_additional_reward
            
        return reward, components

    def step(self, action):
        """Step environment, modify reward, and return observation."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
