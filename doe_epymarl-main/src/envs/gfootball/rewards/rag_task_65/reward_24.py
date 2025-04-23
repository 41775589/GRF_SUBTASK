import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense shooting and passing scenario-based reward.
    It encourages scenarios where agents focus on enhancing precision, 
    decision-making, and strategic positioning.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_skill_improvement_reward_factor = 0.1
        self.shooting_skill_improvement_reward_factor = 0.2

    def reset(self):
        """
        Reset the environment and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Modify rewards based on precision and strategic execution.

        Args:
            reward (list[float]): List of rewards for each player.
        
        Returns:
            tuple[list[float], dict[str, list[float]]]
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward),
                      "possession_continue_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage passing when controlling the ball and teammates are better positioned
            if ('ball_owned_team' in o and o['ball_owned_team'] == 1 and
                    'right_team' in o):
                positioning_scores = [np.minimum(1.0, o['right_team'][i][0]) for i in range(len(o['right_team']))]
                best_position = max(positioning_scores)
                current_position = positioning_scores[o['active']]
                if current_position < best_position:
                    components["positional_reward"][rew_index] = self.passing_skill_improvement_reward_factor
                    reward[rew_index] += self.passing_skill_improvement_reward_factor

            # Reward for shooting attempts close to goal
            if 'ball' in o and o['ball'][0] > 0.75:
                components["possession_continue_reward"][rew_index] = self.shooting_skill_improvement_reward_factor
                reward[rew_index] += self.shooting_skill_improvement_reward_factor

        return reward, components

    def step(self, action):
        """
        Execute a step using the underlying environment, then modify the rewards 
        using customized reward scenarios, and track sticky actions.
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
