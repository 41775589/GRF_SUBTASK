import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive play and efficient ball distribution in a football game."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize metrics to encourage defensive coordination and ball control
        self.ball_recovery_reward = 0.2
        self.effective_pass_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        state = self.env.set_state(from_pickle)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        defensive_reward = [0.0] * len(reward)
        passing_reward = [0.0] * len(reward)

        for idx, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] != -1:
                # Reward for ball recovery
                defensive_reward[idx] += self.ball_recovery_reward
            
            if 'action' in o and o['action'] in ['pass', 'long_pass']:
                # Reward for effective passing
                passing_reward[idx] += self.effective_pass_reward

            reward[idx] += defensive_reward[idx] + passing_reward[idx]

        components['defensive_reward'] = defensive_reward
        components['passing_reward'] = passing_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Include all reward components in the info for monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions for analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_flag
                
        return observation, reward, done, info
