import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a stamina conservation reward encouraging strategic use of Stop-Sprint and Stop-Moving actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
        components = {
            "base_score_reward": reward.copy(),
            "stamina_conserve_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage sporadic use of sprint and active movement towards stamina conservation
            # Use the sticky_actions index for sprint (8) and dribble (9), analyzing their transition from 1 to 0
            previous_sticky = self.sticky_actions_counter[rew_index]
            current_sticky = o['sticky_actions'][8]  # Sprint action index
            if previous_sticky > 0 and current_sticky == 0:
                # Reward for stopping the sprint action
                components["stamina_conserve_reward"][rew_index] = 1.0
            elif current_sticky > 0:
                # Penalize continued use of sprint action
                components["stamina_conserve_reward"][rew_index] = -0.01 * current_sticky
            
            self.sticky_actions_counter[rew_index] = current_sticky

        # Update rewards with stamina conservation components
        for i in range(len(reward)):
            reward[i] += components["stamina_conserve_reward"][i]

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
            
        return observation, reward, done, info
