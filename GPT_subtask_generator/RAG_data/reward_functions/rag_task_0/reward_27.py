import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for focusing on offensive strategies including accurate shooting, 
    dribbling, and different pass types."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize parameters for specific checkpoints and rewards
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05
        self.dribble_reward = 0.05
        self.shot_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["pass_reward"][rew_index] = 0
            components["dribble_reward"][rew_index] = 0
            components["shot_reward"][rew_index] = 0

            if o['game_mode'] in [0]:  # Normal play
                # Encourage passing, dribbling, and shooting
                if 'long_pass' in o['sticky_actions'] and o['sticky_actions']['long_pass']:
                    components["pass_reward"][rew_index] = self.pass_reward
                if 'high_pass' in o['sticky_actions'] and o['sticky_actions']['high_pass']:
                    components["pass_reward"][rew_index] += self.pass_reward
                if 'short_pass' in o['sticky_actions'] and o['sticky_actions']['short_pass']:
                    components["pass_reward"][rew_index] += self.pass_reward / 2  # less reward for short passes
                
                if o['sticky_actions']['dribble']:
                    components["dribble_reward"][rew_index] = self.dribble_reward

                if 'shot' in o['sticky_actions'] and o['sticky_actions']['shot']:
                    components["shot_reward"][rew_index] = self.shot_reward

            # Calculate total modified reward
            reward[rew_index] += (components["pass_reward"][rew_index] +
                                  components["dribble_reward"][rew_index] +
                                  components["shot_reward"][rew_index])

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
