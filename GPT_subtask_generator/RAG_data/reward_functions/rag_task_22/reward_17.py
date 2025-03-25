import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Wrapper adds rewards for effective sprint activation and defensive coverage."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_rewards = 0.05  # Sprint should be rewarded slightly on each step it is activated

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions'])
        return from_pickle
    
    def reward(self, reward):
        # Collect rewards and add component rewards to base agent's reward
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": [0.0] * len(reward)
        }

        # Fallback condition to handle scenarios where observations do not load adequately
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Reward agents for using the sprint action effectively in movement
        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            sprint_action = o['sticky_actions'][8]  # Index 8 corresponds to sprint action
            self.sticky_actions_counter[8] += sprint_action

            # Incremental reward for sprint usage
            if sprint_action:
                components["sprint_reward"][rew_index] = self.sprint_rewards * self.sticky_actions_counter[8]
            
            # Modify reward based on computed components
            reward[rew_index] += components["sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Adding each component of reward to the info dict for tracking
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
