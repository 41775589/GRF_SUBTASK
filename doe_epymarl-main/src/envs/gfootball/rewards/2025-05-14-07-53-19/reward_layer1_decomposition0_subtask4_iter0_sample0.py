import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a passive defensive techniques reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Tracks when the player uses "Stop Moving" and "Stop Sprint" actions effectively.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()
        tactical_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {'base_score_reward': base_score_reward, 'tactical_reward': tactical_reward}

        for idx in range(len(reward)):
            o = observation[idx]
            # Increment tactical points for using "Stop Moving" effectively when close to the own goal
            if o['sticky_actions'][7] == 1:  # Assuming 'Stop Moving' is at index 7
                if np.linalg.norm(o['right_team'][o['active']] - [1, 0]) > 0.8:  # Close to own goal on right side
                    tactical_reward[idx] += 0.05
            # Increment for "Stop Sprint" when pacing is important near midfield
            if o['sticky_actions'][8] == 1:  # Assuming 'Stop Sprint' is at index 8
                if 0.3 < np.linalg.norm(o['ball']) < 0.7:
                    tactical_reward[idx] += 0.03

            # Combine rewards
            reward[idx] += tactical_reward[idx]

        return reward, {'base_score_reward': base_score_reward, 'tactical_reward': tactical_reward}

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
