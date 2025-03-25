import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive task-focused reward based on interceptions and quick counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self._num_defensive_checkpoints = 5  # Number of zones for defense
        self._num_counterattack_checkpoints = 5  # Number of zones for counter-attack
        self._defensive_reward = 0.05
        self._counterattack_reward = 0.1
        self._collected_checkpoints = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "counterattack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_owned_team = o.get('ball_owned_team', -1)
            ball_owned_player = o.get('ball_owned_player', -1)
            active_player = o.get('active', -1)

            if ball_owned_team == o['team'] and ball_owned_player == active_player:
                # Defensive reward layer
                if o['game_mode'] in [1]:  # Braking from opponent's play
                    defensive_checkpoint = int(o['ball'][0] // 0.2) + 1
                    if self._collected_checkpoints.get(rew_index, set()).isdisjoint({defensive_checkpoint}):
                        components["defensive_reward"][rew_index] = self._defensive_reward
                        self._collected_checkpoints[rew_index] = self._collected_checkpoints.get(rew_index, set())
                        self._collected_checkpoints[rew_index].add(defensive_checkpoint)
                    
                # Counter-Attack reward layer
                if o['game_mode'] in [4, 5]:  # Preparing or executing a counter-attack
                    counterattack_checkpoint = int(o['ball'][0] // -0.2) + 1
                    if counterattack_checkpoint > 0 and self._collected_checkpoints.get(rew_index, set()).isdisjoint({counterattack_checkpoint}):
                        components["counterattack_reward"][rew_index] = self._counterattack_reward
                        self._collected_checkpoints[rew_index] = self._collected_checkpoints.get(rew_index, set())
                        self._collected_checkpoints[rew_index].add(counterattack_checkpoint)

            reward[rew_index] += components["defensive_reward"][rew_index] + components["counterattack_reward"][rew_index]

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
