import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for learning offensive skills including
    passing, shooting, dribbling, and sprinting to create scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.rewards_components = {
            "base_score_reward": 0.0,
            "pass_bonus": 0.05,
            "shoot_bonus": 0.1,
            "dribble_bonus": 0.02,
            "sprint_bonus": 0.01
        }
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # no specific state to restore from pickle as of now
        return from_pickle

    def reward(self, reward):
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observations is None:
            return reward, components
        
        assert len(reward) == len(observations)
        
        for i in range(len(reward)):
            observation = observations[i]
            active_actions = observation.get('sticky_actions', [0] * 10)
            self.sticky_actions_counter += active_actions
            components['pass_bonus'] = (active_actions[0] + active_actions[1]) * self.rewards_components['pass_bonus']
            components['shoot_bonus'] = active_actions[3] * self.rewards_components['shoot_bonus']
            components['dribble_bonus'] = active_actions[9] * self.rewards_components['dribble_bonus']
            components['sprint_bonus'] = active_actions[8] * self.rewards_components['sprint_bonus']

            total_bonus = sum([
                components['pass_bonus'], 
                components['shoot_bonus'], 
                components['dribble_bonus'],
                components['sprint_bonus']
            ])
            
            reward[i] += total_bonus

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info["final_reward"] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, new_reward, done, info
