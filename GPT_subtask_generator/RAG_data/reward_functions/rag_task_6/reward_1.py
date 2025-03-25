import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on the proficient usage of Stop-Sprint and Stop-Moving actions for energy conservation."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Specific actions indices in the sticky_actions array
        # Assumes Sprint(8) and Dribble(9) as the potentially restful actions
        self.stop_moving_index = 0  # Assuming 'action_left' as stopping movement (placeholder)
        self.stop_sprinting_index = 8  # Action index for stopping sprinting

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "energy_conservation_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check sticky actions for stop moving and stop sprinting
            sticky_actions = o['sticky_actions']
            no_sprint = sticky_actions[self.stop_sprinting_index] == 0
            stop_moving = sticky_actions[self.stop_moving_index] == 0

            # Reward for not sprinting while also not moving to conserve energy
            if no_sprint and stop_moving:
                components["energy_conservation_reward"][rew_index] = 0.05  # Small reward for energy conservation

            # Update the final reward
            reward[rew_index] += components["energy_conservation_reward"][rew_index]

        return reward, components

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
