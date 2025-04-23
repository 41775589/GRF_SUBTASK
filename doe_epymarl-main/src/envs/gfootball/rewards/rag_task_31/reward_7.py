import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on defensive maneuvers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky actions counter

    def reset(self):
        """ Reset for a new episode """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Save state for checkpointing. """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restore state from checkpoint. """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """ Modify reward based on defensive actions. """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": np.zeros(len(reward)),
            "slide_reward": np.zeros(len(reward))
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Calculate rewards for tackles and slides
            if 'sticky_actions' in o:
                # action indices corresponding to tackle and slide might vary, check environment specifics
                tackle_action = 6  # hypothetical index for tackle action
                slide_action = 7   # hypothetical index for slide action
                
                # Increment counters and apply rewards if these actions are taken
                if o['sticky_actions'][tackle_action] == 1:
                    self.sticky_actions_counter[tackle_action] += 1
                    components["tackle_reward"][rew_index] = 0.5  # positive reward for successful tackle
                    
                if o['sticky_actions'][slide_action] == 1:
                    self.sticky_actions_counter[slide_action] += 1
                    components["slide_reward"][rew_index] = 0.3  # positive reward for successful slide

                # Update reward
                reward[rew_index] += components["tackle_reward"][rew_index] + components["slide_reward"][rew_index]

        return reward, components

    def step(self, action):
        """ Execute environment step and augment with defensive reward adjustments. """
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
