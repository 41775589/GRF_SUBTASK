import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward component for shooting skills from a distance, integrating positioning and open shots against defenders."""
    
    def __init__(self, env):
        super().__init__(env)
        self.min_shoot_distance = 0.5  # Threshold for considering a shoot to be long range
        self.shoot_reward = 0.5  # Reward for shooting from outside this range
        self.distance_penalty = 0.1  # Penalty for loosing ball too close to own goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any state if necessary, not needed currently.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "shoot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the ball is far enough and player had a shooting opportunity
            if (o['ball_owned_team'] == 1 and np.linalg.norm(o['ball'][:2]) > self.min_shoot_distance):
                # Assuming positive reward for goal attempt from long distance
                components["shoot_reward"][rew_index] = self.shoot_reward
                reward[rew_index] += components["shoot_reward"][rew_index]

            # Penalty if losing the ball close to own goal due to a bad long shot
            if (o['ball_owned_team'] == 0 and np.linalg.norm(o['ball'][:2]) < -0.5):
                components["distance_penalty"] = -self.distance_penalty
                reward[rew_index] += components["distance_penalty"]

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
