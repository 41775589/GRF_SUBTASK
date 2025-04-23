import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on promoting technical skills for high passes in the football game."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_booster = 0.20  # Reward boost for successful high passes
        self._for_own_goal = [0.99, -0.99]  # Target goal positions for high passes (right and left sides respectively)

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
        # Initialize the components dictionary
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o is not None:
                # High pass execution check, action 6 corresponds to 'action_high_pass'
                if ('ball' in o and 'ball_direction' in o and 
                    o['sticky_actions'][6] == 1 and 
                    np.linalg.norm(o['ball_direction'][:2]) > 0.05):  # Check if the ball direction is strong enough
                    target_goal_y = self._for_own_goal[o['ball_owned_team']]
                    ball_is_high = o['ball'][2] > 0.15  # Check if the ball Z coordinate indicates a high pass
                    ball_heading_to_goal = (np.sign(o['ball'][0]) == np.sign(target_goal_y))

                    if ball_is_high and ball_heading_to_goal:
                        components["high_pass_reward"][rew_index] = self._high_pass_booster
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
