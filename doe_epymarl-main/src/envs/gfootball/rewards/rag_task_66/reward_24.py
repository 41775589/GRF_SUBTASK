import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on effective passing under defensive pressure, to enhance ball retention."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_bonus = 0.1
        self.defensive_pressure_threshold = 0.2
        self.pass_count = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_count = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned = (o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player'])
            
            # Only reward passes if under defensive pressure and possession maintained
            if ball_owned and any(np.linalg.norm(o['left_team'][i] - o['ball'][:2]) <= self.defensive_pressure_threshold for i in range(len(o['left_team']))):
                if 'action_bottom' in o['sticky_actions'] and o['sticky_actions']['action_bottom']:
                    self.pass_count += 1

            # Calculate and accumulate rewards
            components["pass_reward"][rew_index] = self.pass_count * self.pass_bonus
            reward[rew_index] += components["base_score_reward"][rew_index] + components["pass_reward"][rew_index]

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

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {"pass_count": self.pass_count}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_count = from_pickle['CheckpointRewardWrapper']['pass_count']
        return from_pickle
