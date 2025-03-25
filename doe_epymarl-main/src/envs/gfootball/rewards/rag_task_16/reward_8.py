import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes accurately."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._pass_accuracy_checkpoints = {}
        self._max_checkpoints = 5
        self._checkpoint_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_accuracy_checkpoints = {}
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._pass_accuracy_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._pass_accuracy_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' not in o or o['ball_owned_team'] == -1:
                continue

            if o['ball_owned_team'] == 0 and 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                # High pass made
                if any(action == 7 for action in o['sticky_actions']):
                    # Calculate accuracy rewards based on the trajectory control and situational efficiency
                    player_pos = o['left_team'][o['active']]
                    ball_pos = o['ball'][:2]
                    distance_to_target = np.linalg.norm(ball_pos - player_pos)
                    normalized_distance = min(1, distance_to_target / 100)
                    components["pass_accuracy_reward"][rew_index] += self._checkpoint_reward * (1 - normalized_distance)
                    reward[rew_index] += components["pass_accuracy_reward"][rew_index]
                    self._pass_accuracy_checkpoints[rew_index] = min(
                        self._max_checkpoints, self._pass_accuracy_checkpoints.get(rew_index, 0) + 1)
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
