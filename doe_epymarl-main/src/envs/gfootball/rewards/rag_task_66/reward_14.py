import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for mastering short passing under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Divide the field into 5 zones for controlled ball passing
        self._completion_reward = 0.2
        self._zone_threshold = 0.2  # Distance threshold to consider passing successful
        self._passes_completed = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self._passes_completed = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._passes_completed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._passes_completed = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, ob in enumerate(observation):
            current_player = ob.get('active')
            controlled_player_pos = ob['right_team'][current_player] if ob['ball_owned_team'] == 1 else ob['left_team'][current_player]
            if not self._passes_completed.get(i):
                self._passes_completed[i] = [False] * self._num_zones
            
            # Check the position of the ball to determine the zone
            ball_pos = ob['ball'][:2]
            zone_idx = min(int((ball_pos[0] + 1) // (2 / self._num_zones)), self._num_zones - 1)

            if not self._passes_completed[i][zone_idx]:
                # Check if the ball is owned by the controlled team and within the short passing threshold
                if ob['ball_owned_team'] == ob['ball_owned_player'] and np.linalg.norm(ball_pos - controlled_player_pos) <= self._zone_threshold:
                    # Pass completed in this zone
                    self._passes_completed[i][zone_idx] = True
                    components["pass_completion_reward"][i] = self._completion_reward
                    reward[i] += components["pass_completion_reward"][i]

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
