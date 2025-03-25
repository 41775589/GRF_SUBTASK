import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for defensive coordination near the penalty area."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.covered_areas = np.zeros(2, dtype=bool)  # Tracker for defensive covering

    def reset(self):
        """
        Reset the environment and clear trackers for new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.covered_areas.fill(False)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of the environment and the state of the reward wrapper.
        """
        to_pickle['covered_areas'] = self.covered_areas.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the environment and the state of the reward wrapper.
        """
        from_pickle = self.env.set_state(state)
        self.covered_areas = from_pickle['covered_areas']
        return from_pickle

    def reward(self, reward):
        """
        Calculate and return modified reward based on defensive position and coverage.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'defense_coverage_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, (obs, rew) in enumerate(zip(observation, reward)):
            defensive_area_near_penalty = np.linalg.norm(obs['left_team'][:, :2] + np.array([1, 0]), axis=1) < 0.1
            covered_this_step = defensive_area_near_penalty.any()
            
            # Reward agents for being in important defensive areas
            if not self.covered_areas[index] and covered_this_step:
                components['defense_coverage_reward'][index] = 0.5  # Reward for first cover
                reward[index] += components['defense_coverage_reward'][index]
                self.covered_areas[index] = True

        return reward, components

    def step(self, action):
        """
        Environment step wrapped with the custom reward assessment.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Add sticky actions tracking for debugging and learning
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
                info[f"sticky_actions_{i}"] = action_state

        return observation, reward, done, info
