import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for an agent acting like a hybrid midfielder/defender."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._target_action_rewards = {
            8: 0.02,  # Sprint
            0: -0.02,  # Stop Sprint
            9: 0.03,  # Dribble
            11: 0.05,  # High Pass
            10: 0.05   # Long Pass
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_counter']
        return from_pickle

    def reward(self, reward):
        """ Adjust the reward based on specific actions to encourage the desired behaviors."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "action_rewards": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']
            
            # Add rewards for specific actions being active
            action_reward = sum([self._target_action_rewards.get(idx, 0) * act 
                                 for idx, act in enumerate(sticky_actions)])
            components["action_rewards"][rew_index] = action_reward
            reward[rew_index] += action_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                if 'sticky_actions' in agent_obs:
                    for i, active in enumerate(agent_obs['sticky_actions']):
                        self.sticky_actions_counter[i] += active
                        info[f"sticky_actions_{i}"] = active
        return observation, reward, done, info
