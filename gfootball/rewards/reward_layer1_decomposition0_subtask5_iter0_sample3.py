import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes defensive actions and quick response to opponent's attacks
    by giving a dense reward for tackling, intercepting, or quickly retrieving the ball.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_action_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            defensive_bonus = 0

            # Rewards for intercepting the ball or tackling if ball is not currently possessed
            if o['ball_owned_team'] == -1:
                ball_dist = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']])
                if ball_dist < 0.05:  # if very close to the ball
                    defensive_bonus += 0.3

            elif o['ball_owned_team'] == 1:  # opponent controls the ball
                player_pos = o['left_team'][o['active']]
                opponent_pos = o['right_team'][o['ball_owned_player']]
                dist_to_opponent = np.linalg.norm(player_pos - opponent_pos)

                # Rewards for getting close to the ball when the opponent possesses it
                if dist_to_opponent < 0.1:
                    defensive_bonus += 0.5

            # Update reward if there was a defensive action
            components["defensive_action_reward"][rew_index] = defensive_bonus
            reward[rew_index] += defensive_bonus

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
