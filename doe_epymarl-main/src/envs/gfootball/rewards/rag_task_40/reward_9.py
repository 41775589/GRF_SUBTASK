import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for successful defensive actions and strategic counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["defensive_reward"][rew_index] = 0

            # Reward intercepting the ball
            if o.get('ball_owned_team') == 0:
                if o.get('game_mode') in [2, 3, 4, 5]:  # Defensive game modes
                    components["defensive_reward"][rew_index] += 0.1

            # Reward for recovering ball in personal half and passing forward successfully
            if o.get('ball_owned_team') == 0 and o['ball'][0] < 0:  # Ball in defensive half
                next_obs, _, _, _ = self.env.step([5])  # Emulate pass action forward
                if next_obs[rew_index]['ball_owned_team'] == 0 and next_obs[rew_index]['ball'][0] > 0:
                    components["defensive_reward"][rew_index] += 0.5
                self.env.undo()  # Revert emulation for consistent state

            # Accumulate custom rewards with existing reward
            reward[rew_index] += components["defensive_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action_value in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action_value
        return observation, reward, done, info
