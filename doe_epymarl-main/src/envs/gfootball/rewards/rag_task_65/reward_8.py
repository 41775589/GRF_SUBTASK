import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward function by emphasizing shooting and passing accuracy
       under various strategic positions and game contexts."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_strategic_points = 5
        self._strategic_point_reward = 0.05

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Enhance the incoming rewards by rewarding precision in passing and shooting,
           strategic decision-making, based on the position and context of the game."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "strategic_point_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            # Reward players for shooting near strategic positions or successfully passing under pressure
            if obs['game_mode'] in (2, 6):  # Modes related to shooting: FreeKick and Penalty
                components["strategic_point_reward"][rew_index] = self._strategic_point_reward * self._num_strategic_points
                reward[rew_index] += components["strategic_point_reward"][rew_index]

            elif obs['game_mode'] == 4:  # Modes related to passing: Corner
                distance_to_goal = np.linalg.norm([obs['ball'][0] - 1, obs['ball'][1]])
                if distance_to_goal < 0.2:  # Ball very close to the goal from corner
                    components["strategic_point_reward"][rew_index] += 0.1  # Higher reward for tighter situations
                reward[rew_index] += components["strategic_point_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Take a step using the given action, modify the rewards and return step outputs."""
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

    def get_state(self, to_pickle):
        """Get the state of the environment with additional wrapper state if any."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment with additional wrapper state if any."""
        from_pickle = self.env.set_state(state)
        return from_pickle
