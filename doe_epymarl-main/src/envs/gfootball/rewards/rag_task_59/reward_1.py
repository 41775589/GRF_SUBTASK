import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for goalkeeper coordination in high-pressure scenarios."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positive_reinforcement = 0.1
        self.negative_reinforcement = -0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        for rew_idx in range(len(reward)):
            o = observation[rew_idx]
            components.setdefault("positioning_reward", [0.0] * len(reward))
            components.setdefault("clearance_reward", [0.0] * len(reward))

            # Specialized positioning reward for goalkeeper under pressure
            if o['game_mode'] in [2, 3, 4, 5] and o['active'] == o['designated']:
                if o['active'] == np.argmin(o['left_team'][:, 0]):
                    components["positioning_reward"][rew_idx] = self.positive_reinforcement
            else:
                components["positioning_reward"][rew_idx] = self.negative_reinforcement
            
            # Reward for clearing the ball correctly to minimize goal threats
            if o['game_mode'] == 2:  # Goal kick situation
                if o['ball_direction'][0] > 0 and o['left_team_roles'][o['active']] == 0:  # Goalkeeper role is 0
                    distance_to_closest_defender = min(np.linalg.norm(o['left_team'][:, :2] - o['ball'][:2], axis=1))
                    if distance_to_closest_defender < 0.5:  # Approximately the half of the field
                        components["clearance_reward"][rew_idx] = self.positive_reinforcement

            # Compile the total reward for the current index
            reward[rew_idx] += (components["positioning_reward"][rew_idx] +
                                components["clearance_reward"][rew_idx])
        
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
