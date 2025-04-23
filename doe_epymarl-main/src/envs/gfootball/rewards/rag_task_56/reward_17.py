import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focusing on enhancing defensive capabilities in football simulation."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._goalkeeper_efficiency = 0.5
        self._defender_efficiency = 0.3

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get state to save - no specific state to add for wrapper."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from loaded data."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on defensive actions and success."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_efficiency": [0.0] * len(reward),
            "defender_efficiency": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage goalkeeper actions: effective shot stopping
            if o['active'] == 0 and o['score'][1] == 0:
                if o['ball_owned_player'] == o['active'] and o['game_mode'] in [3, 6]:
                    components['goalkeeper_efficiency'][rew_index] = self._goalkeeper_efficiency
                    reward[rew_index] += components['goalkeeper_efficiency'][rew_index]

            # Encourage defenders: good positioning and ball recovery
            if o['active'] in [1, 2, 3, 4] and o['ball_owned_team'] == 0:
                distance_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']][:2])
                if distance_to_ball < 0.1:  # effectively close to the ball
                    components['defender_efficiency'][rew_index] = self._defender_efficiency
                    reward[rew_index] += components['defender_efficiency'][rew_index]

        return reward, components

    def step(self, action):
        """Execute one time step within the environment."""
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
