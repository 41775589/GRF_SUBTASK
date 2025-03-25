import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on the technical aspects and precision of long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.completed_passes = 0
        self.total_passes = 0  # Track the total number of pass attempts
        # Reward related parameters
        self.pass_accuracy_reward = 50
        self.pass_distance_reward_coef = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.completed_passes = 0
        self.total_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['completed_passes'] = self.completed_passes
        to_pickle['total_passes'] = self.total_passes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.completed_passes = from_pickle.get('completed_passes', 0)
        self.total_passes = from_pickle.get('total_passes', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {
            "base_score_reward": reward,
            "pass_accuracy_reward": [0],
            "pass_distance_reward": [0]
        }
        
        for rew_index, o in enumerate(observation):
            ball_position = o['ball']
            ball_owned_team = o.get('ball_owned_team', -1)
            # Check pass completion
            if ball_owned_team != -1:
                self.total_passes += 1
                ball_direction = o['ball_direction']
                distance = np.linalg.norm(ball_direction[:2])  # consider x, y only
                if distance > 0.5:
                    self.completed_passes += 1
                    components["pass_accuracy_reward"][0] = self.pass_accuracy_reward
                    components["pass_distance_reward"][0] = self.pass_distance_reward_coef * distance
            
            # Calculate overall reward
            reward[rew_index] += components["pass_accuracy_reward"][0] + components["pass_distance_reward"][0]

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
