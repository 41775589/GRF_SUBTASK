import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes offensive football actions like passing, shooting, and dribbling."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize counters for specific action rewards
        self.pass_count = 0
        self.shot_count = 0
        self.dribble_count = 0
        self.sprint_count = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.pass_count = 0
        self.shot_count = 0
        self.dribble_count = 0
        self.sprint_count = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_count'] = self.pass_count
        to_pickle['shot_count'] = self.shot_count
        to_pickle['dribble_count'] = self.dribble_count
        to_pickle['sprint_count'] = self.sprint_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_count = from_pickle.get('pass_count', 0)
        self.shot_count = from_pickle.get('shot_count', 0)
        self.dribble_count = from_pickle.get('dribble_count', 0)
        self.sprint_count = from_pickle.get('sprint_count', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": np.array(reward).copy(),
            "pass_reward": 0.0,
            "shot_reward": 0.0,
            "dribble_reward": 0.0,
            "sprint_reward": 0.0
        }

        if not observation:
            return reward, components

        for agent_obs in observation:
            if agent_obs['sticky_actions'][7] and self.pass_count < 5:
                # Reward for passing based on sticky action index for pass
                components["pass_reward"] += 0.1
                self.pass_count += 1

            if agent_obs['sticky_actions'][2] and self.shot_count < 3:
                # Reward for shooting based on sticky action index for shoot
                components["shot_reward"] += 0.2
                self.shot_count += 1

            if agent_obs['sticky_actions'][9] and self.dribble_count < 10:
                # Reward for dribbling based on sticky action index for dribble
                components["dribble_reward"] += 0.05
                self.dribble_count += 1

            if agent_obs['sticky_actions'][8] and self.sprint_count < 20:
                # Reward for sprinting based on sticky action index for sprint
                components["sprint_reward"] += 0.03
                self.sprint_count += 1
            
        total_rewards = reward + sum(components.values())
        return total_rewards, components

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
