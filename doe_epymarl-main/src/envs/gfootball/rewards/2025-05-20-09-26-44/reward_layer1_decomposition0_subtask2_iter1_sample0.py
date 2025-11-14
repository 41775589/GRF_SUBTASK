import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards to defensive actions such as sliding and stopping, with an emphasis on precision under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self._num_agents = 1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_action = None
        
        # Introducing an improved decay factor to gradually reduce reward over continuous use to prevent spamming
        self.decay_factor = 0.9
        self.reward_scale = 0.5
    
    def reset(self):
        """Reset the state at the beginning of each episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_action = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of inner environment as well as the local counter state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Ensure that state restoration includes our sticky action counters."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Customize reward based on defensive effort by agents."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_effort_reward": [0.0]}

        for i, o in enumerate(observation):
            slide_action = o['sticky_actions'][7]  # 7 might correspond to sliding or significant defensive action
            stop_action = o['sticky_actions'][6]  # 6 might be ideally corresponding to stopping action
            
            if (slide_action != self.last_action or stop_action != self.last_action) and (slide_action or stop_action):
                # We provide a positive reward for initiating the action and reduce it based on subsequent repeated action
                if slide_action == 1:
                    components['defensive_effort_reward'][i] += self.reward_scale
                if stop_action == 1:
                    components['defensive_effort_reward'][i] += self.reward_scale * 0.5  # stop is less valued than slide

                components['defensive_effort_reward'][i] *= self.decay_factor ** self.sticky_actions_counter[7]
                self.sticky_actions_counter[7] += 1
            
            self.last_action = slide_action if slide_action else stop_action
            
            # Update the total reward with the additional component.
            reward[i] += components['defensive_effort_reward'][i]

        return reward, components

    def step(self, action):
        """Wrap environment's step to include detailed reward calculation."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
