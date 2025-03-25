import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on enhancing defensive strategies by incentivizing rapid repositioning 
    and increased use of sprint in defensive scenarios.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = dict(sticky_actions_counter=self.sticky_actions_counter.tolist())
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle.get('CheckpointRewardWrapper', {}).get('sticky_actions_counter', np.zeros(10, dtype=int)))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, obs in enumerate(observation):
            sprint_action_active = obs['sticky_actions'][8]  # Index 8 is 'action_sprint'
            if sprint_action_active:
                # Increase reward if sprinting while a defensive mode is active (like game_mode indicates FreeKick Against)
                if obs['game_mode'] in {2, 3, 4}:  # FreeKick, GoalKick, or Corner
                    components['sprint_reward'][rew_index] += 0.05
                    reward[rew_index] += 0.05 * self.sticky_actions_counter[8]  # Increment if action 'sprint' used more frequently
                    self.sticky_actions_counter[8] += 1  # Counting sprint usage for tuning potential
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
