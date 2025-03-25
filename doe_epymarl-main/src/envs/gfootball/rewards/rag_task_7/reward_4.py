import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to emphasize learning sliding tackles in defense, specifically focusing on timing
    and precision under pressure using proximity to the opponent who has the ball.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_reward = 1.0
        self._pressure_threshold = 0.1  # Proximity threshold for high pressure
        self._defensive_success_bonus = 2.0  # Bonus for successful tackle

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Adjust reward based on defensive actions
            if o['sticky_actions'][9] == 1:  # Action dribble as placeholder for tackle in this example
                opposing_player_dist = np.min(np.linalg.norm(o['right_team'] - o['ball'], axis=1))
                if opposing_player_dist < self._pressure_threshold:
                    # Increase rewards for actions when very close to ball possession under pressure
                    components["tackling_reward"][rew_index] = self._tackle_reward
                    if o['ball_owned_team'] == 1 and o['ball_owned_player'] == np.argmin(opposing_player_dist):
                        components["tackling_reward"][rew_index] += self._defensive_success_bonus
                reward[rew_index] += components["tackling_reward"][rew_index]

        components["final_reward"] = sum(reward)
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
