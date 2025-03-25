import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for tactical defensive training and transition to counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._defensive_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defensive_positions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o.get('ball_owned_team', -1)
            ball_owned_player = o.get('ball_owned_player', -1)
            game_mode = o.get('game_mode', 0)

            # Rewarding defensive actions specially to foster quick transitions
            if ball_owned_team == 0 and ball_owned_player == o['active']:
                if 0 <= o['ball'][0] <= 0.2:  # If ball is on the defensive half
                    components['defensive_reward'][rew_index] = 0.05
                    reward[rew_index] += components['defensive_reward'][rew_index]
                self._defensive_positions[rew_index] = o['ball'][0]

            # Rewards transitions to counter-attacks
            if game_mode == 0 and ball_owned_team == 0:  # Normal gameplay
                previous_position = self._defensive_positions.get(rew_index, 0)
                if previous_position >= 0.2 and o['ball'][0] > previous_position:
                    # Encourage moving the ball forward rapidly from a defensive stance
                    transition_reward = 0.1 * (o['ball'][0] - previous_position)
                    components['defensive_reward'][rew_index] += transition_reward
                    reward[rew_index] += transition_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
