import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward encouraging successful sliding tackles near the defensive third."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_reward = 0.5  # Reward for successfully tackling in the defensive third
        self.defensive_third_threshold = -0.35  # X-coordinate boundary defining the defensive third

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward

        # Apply the reward function logic based on tackle effectiveness
        for idx, o in enumerate(observation):
            components.setdefault("tackle_reward", [0.0, 0.0])
            
            # Check if ball is in defensive third and if a tackle or interception occurs
            if o['ball'][0] <= self.defensive_third_threshold and 'sticky_actions' in o:
                successful_tackle = o['sticky_actions'][8]  # assuming index 8 is tackle
                if successful_tackle and o['ball_owned_team'] != o['active']:
                    components["tackle_reward"][idx] = self.tackle_success_reward
                    reward[idx] += components["tackle_reward"][idx]

        return reward, components

    def step(self, action):
        # This method should not be changed according to instructions
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
