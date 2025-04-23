import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on controlled passing and shooting accuracy as well as strategic positioning."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize game action parameters influencing rewards
        self.pass_rewards = 0.05
        self.shoot_rewards = 0.1
        self.position_bonus = 0.02
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        # Ensure the output format is correct
        assert len(reward) == len(observation)
        components['pass_reward'] = [0.0] * len(reward)
        components['shoot_reward'] = [0.0] * len(reward)
        components['position_reward'] = [0.0] * len(reward)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if there's an active shooting action
            if 'shoot' in o['sticky_actions']:
                components['shoot_reward'][rew_index] = self.shoot_rewards
                reward[rew_index] += components['shoot_reward'][rew_index]

            # Check if there's a passing action:
            if 'pass' in o['sticky_actions']:
                components['pass_reward'][rew_index] = self.pass_rewards
                reward[rew_index] += components['pass_reward'][rew_index]

            # Strategic positioning rewards based on being close to tactical zones
            if o['position'] in strategic_positions(self):
                components['position_reward'][rew_index] = self.position_bonus
                reward[rew_index] += components['position_reward'][rew_index]

        return reward, components

    def strategic_positions(self):
        # Define strategic positions dynamically if needed based on in-game conditions
        return [(1, 0.5), (-1, -0.5)]   # Simplified example positions

    def step(self, action):
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
