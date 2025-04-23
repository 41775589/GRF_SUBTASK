import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on advanced ball control and passing skills under high-pressure scenarios.
    Specifically, this wrapper incentivizes short, high, and long passes when under tight defensive pressure.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Control reward is incremented whenever a successful pass is made under pressure
        self.pass_under_pressure_reward = 0.5
        self.passing_skill_multiplier = {
            0: 0.5,   # action_short_pass
            1: 1.0,   # action_high_pass
            4: 1.5    # action_long_pass
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': [reward[i] for i in range(len(reward))],
                      'pass_under_pressure_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Determine if the player is under pressure
            if 'right_team_tired_factor' in o and 'left_team_direction' in o:
                opponents_distance = np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1)
                number_of_close_opponents = np.sum(opponents_distance < 0.1)  # Less than 0.1 units away
                sticky_actions = o['sticky_actions']
                
                # Check for passes when under pressure
                if number_of_close_opponents >= 2:  # Arbitrary number suggesting 'pressure'
                    for action_index, is_active in enumerate(sticky_actions):
                        if is_active and action_index in self.passing_skill_multiplier:
                            components['pass_under_pressure_reward'][rew_index] += (
                                self.pass_under_pressure_reward * self.passing_skill_multiplier[action_index]
                            )

            # Combine base reward with additional components
            reward[rew_index] += components['pass_under_pressure_reward'][rew_index]
        
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
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
