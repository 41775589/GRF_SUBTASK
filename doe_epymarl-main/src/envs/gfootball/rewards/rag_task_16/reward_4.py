import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focusing on enhancing technical skills for executing high passes with precision."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_accuracy_threshold = 0.75  # Assuming normalized distance accuracy required
        self.reward_for_successful_pass = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state of this wrapper."""
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper from a pickle object."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Enhance the reward based on high pass execution quality."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, dtype=float).copy(),
                      "high_pass_reward": np.zeros(len(reward), dtype=float)}
        
        if observation is None:
            return reward, components

        # Assuming that each agent's observation includes whether they attempted a high pass
        # and the efficiency or accuracy of the pass
        for idx, o in enumerate(observation):
            if 'high_pass_attempt' in o and o['high_pass_attempt']:
                if ('high_pass_accuracy' in o and o['high_pass_accuracy'] > self.pass_accuracy_threshold):
                    # Reward the agent for a successful high pass
                    components["high_pass_reward"][idx] = self.reward_for_successful_pass
                # Update reward for this agent
                reward[idx] += components["high_pass_reward"][idx]

        return reward, components

    def step(self, action):
        """Step function that integrates the custom rewards defined."""
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
