import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for precision in high passes. This includes rewarding actions
    that lead to successful high passes, considering the ball's trajectory, the pass's strength, 
    and the appropriateness of the situation for making a high pass.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_height = o['ball'][2]  # Z component of the ball's position
            ball_velocity = np.linalg.norm(o['ball_direction'])  # Magnitude of ball's velocity vector.
            in_control = o['ball_owned_team'] == o['active']  # Is the ball controlled by the active team?

            # We check if the scenario is suitable for a high pass:
            if in_control and ball_height > 0.1 and ball_velocity > 0.05:
                # We reward high balls with adequate height and strength when in control.
                components["high_pass_reward"][rew_index] = 0.5
                reward[rew_index] += components["high_pass_reward"][rew_index]

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
