import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for promoting attacking skills and creative offensive play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._goal_distance_reward = 0.1
        self._successful_pass_reward = 0.05
        self._shoot_efficiency = 0.2

    def reset(self):
        """Reset wrapper state and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get state for serialization."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from deserialization."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        """Modify rewards based on gameplay focused on promoting attacking skills."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_distance_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "shoot_efficiency": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for moving closer to the goal with the ball
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'ball' in o:
                goal_distance = abs(o['ball'][0] - 1)  # Distance from right-side goal
                components["goal_distance_reward"][rew_index] = self._goal_distance_reward * (1 - goal_distance)
                reward[rew_index] += components["goal_distance_reward"][rew_index]

            # Reward for successful passes
            if 'action' in o and o['action'] in [football_action_set.action_short_pass,
                                                football_action_set.action_long_pass]:
                components["pass_reward"][rew_index] = self._successful_pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]

            # Reward efficiency when opted to shoot and successful
            if 'action' in o and o['action'] == football_action_set.action_shot:
                if 'score' in o and o['score'][1] > o['score'][0]:  # if score increased for the right team
                    components["shoot_efficiency"][rew_index] += self._shoot_efficiency
                    reward[rew_index] += components["shoot_efficiency"][rew_index]

        return reward, components

    def step(self, action):
        """Execute a step in the environment, collect observation and reward, and add them to info."""
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
