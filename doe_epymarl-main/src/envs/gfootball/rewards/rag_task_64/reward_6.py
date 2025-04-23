import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specialized reward for executing high passes and crosses from various angles and 
    distances to enhance dynamic attacking plays and improve spatial creation for the team.
    """
    def __init__(self, env):
        super().__init__(env)
        self.cross_pass_threshold = 0.5  # Minimal z-coordinate for high passes
        self.distance_reward_scale = 0.1 # Scaling factor for distance reward

        # Initialize counters and thresholds
        self.cross_pass_counter = np.zeros(10, dtype=int)  # Counter for high passes

    def reset(self):
        # Reset the cross pass counter
        self.cross_pass_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['cross_pass_counter'] = self.cross_pass_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.cross_pass_counter = from_pickle['cross_pass_counter']
        return from_pickle

    def reward(self, reward):
        """
        Compute the enhanced reward function by considering high passes and crosses.
        A high pass is identified when the ball's z-coordinate crosses a predefined
        threshold, encouraging verticality in play.
        """

        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": np.array(reward).copy(),
            "cross_pass_reward": np.zeros_like(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball'][2] > self.cross_pass_threshold and o['ball_owned_team'] == 0:
                # Reward for high crosses based on increasing distance and execution by the controlled agent
                if o['designated'] == o['active']:
                    self.cross_pass_counter[rew_index] += 1
                    components["cross_pass_reward"][rew_index] = self.distance_reward_scale * np.linalg.norm(o['ball'][:2])
                    reward[rew_index] += components["cross_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.cross_pass_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
