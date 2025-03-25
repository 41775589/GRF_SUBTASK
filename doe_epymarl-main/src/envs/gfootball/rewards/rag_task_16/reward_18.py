import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for technical skill enhancement, focusing on executing high passes with precision.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.high_pass_threshold = 0.3  # height above which a pass is considered high
        self.dist_to_goal_threshold = 0.5  # proximity to opponent's goal to encourage precise passes
        self.high_pass_reward = 0.2
        self.precise_pass_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # sticky actions counter for all available actions

    def reset(self):
        """
        Reset the environment and clear the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """
        Serialize the Wrapper along with the environment's state.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Deserialize the state into the Wrapper and the environment.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Apply the custom reward logic based on high passes and pass precision.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": 0.0,
                      "precise_pass_reward": 0.0}

        for o in observation:
            if o['ball'][2] >= self.high_pass_threshold:  # Check ball height
                components["high_pass_reward"] += self.high_pass_reward

            ball_position = o['ball'][:2]
            goal_position = [1.0, 0.0]  # Assuming playing on the right side
            distance_to_goal = np.linalg.norm(np.array(ball_position) - np.array(goal_position))
            if distance_to_goal <= self.dist_to_goal_threshold:
                components["precise_pass_reward"] += self.precise_pass_reward

        # Final reward is a combination of base reward and the additional components
        new_reward = reward[0] + components["high_pass_reward"] + components["precise_pass_reward"]
        return [new_reward for _ in reward], components

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
