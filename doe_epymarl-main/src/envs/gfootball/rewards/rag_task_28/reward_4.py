import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focused on enhancing dribbling skills when 
    facing the goalkeeper. It encourages maintaining control of the 
    ball under pressure and executing feints and quick direction changes.
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the counter of sticky actions.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Include the state of the reward wrapper in the pickle state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the reward wrapper from the pickle state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Generate the reward based on dribbling effectiveness near the goalkeeper.

        Args:
            reward (list[float]): Base reward from the environment.

        Returns:
            tuple[list[float], dict[str, list[float]]]: Adjusted reward and detailed components.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            # Incrementally reward close ball control near the goal
            if (o['ball_owned_team'] == 1) and (o['active'] == o['ball_owned_player']):
                x, y = o['ball'][:2]
                ball_close_to_goal = (x >= 0.8) and (abs(y) < 0.2)
                if ball_close_to_goal:
                    # Detect significant actions
                    meaningful_actions = o['sticky_actions'][8] or o['sticky_actions'][9]  # sprint or dribble
                    if meaningful_actions:
                        components["dribbling_reward"][rew_index] = 0.05
                        reward[rew_index] += components["dribbling_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take a step using the given actions, recompute reward and return observations.

        Args:
            action: Actions to be taken.

        Returns:
            tuple: Observations, reward, done flag and additional info from the environment.
        """
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
