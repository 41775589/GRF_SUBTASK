import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards executing long passes accurately over the football field.
    It targets vision, timing, and precision in ball distribution by the agents.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize a counter for actions and their persistence
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coordinates for designated zones for rewarding accurate long passes
        self.pass_targets = [
            (0.75, 0),  # Center right
            (-0.75, 0), # Center left
            (0.5, 0.3),  # Upper right quadrant
            (-0.5, 0.3),  # Upper left quadrant
            (0.5, -0.3), # Lower right quadrant
            (-0.5, -0.3) # Lower left quadrant
        ]
        # Reward given for hitting each target
        self.pass_reward = 0.2
        self.reached_targets = set()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reached_targets = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.reached_targets
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.reached_targets = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            if o['ball_owned_team'] == o['active'] and o['ball_owned_team'] != -1:
                ball_next_x = o['ball'][0] + o['ball_direction'][0]
                ball_next_y = o['ball'][1] + o['ball_direction'][1]
                ball_position = (ball_next_x, ball_next_y)
                
                # Reward for hitting one of the target zones with a long pass
                for target in self.pass_targets:
                    if target not in self.reached_targets and np.linalg.norm(np.subtract(target, ball_position)) < 0.1:
                        components["pass_reward"][rew_index] = self.pass_reward
                        reward[rew_index] += 1.5 * self.pass_reward
                        self.reached_targets.add(target)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky_actions info for each step outputs
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
