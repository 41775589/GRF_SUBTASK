import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on efficiently utilizing possession recovery
    to foster quick decision making and effective counter-attacks.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # To track which checkpoints have been hit
        self._counterattack_checkpoints = {}
        # Number of zones that divide the field to calculate position rewards
        self._num_zones = 10
        self._possession_recovery_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._counterattack_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._counterattack_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._counterattack_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            if o['game_mode'] in [2, 3, 4, 5, 6]:  # Modes that imply possession change triggers
                if o['ball_owned_team'] == 0:  # Left team just got the possession
                    reward[i] += self._possession_recovery_reward
                    components["possession_transition_reward"][i] = self._possession_recovery_reward

                    # Calculate the checkpoints
                    position = np.linalg.norm(o['ball'][:2])  # Use 2D position from the center
                    while (self._counterattack_checkpoints.get(i, 0) < self._num_zones):
                        threshold = 1 - 1 / self._num_zones * self._counterattack_checkpoints.get(i, 0)
                        if position > threshold:
                            break
                        reward[i] += self._possession_recovery_reward / self._num_zones
                        components["possession_transition_reward"][i] += (self._possession_recovery_reward / self._num_zones)
                        self._counterattack_checkpoints[i] = self._counterattack_checkpoints.get(i, 0) + 1

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
