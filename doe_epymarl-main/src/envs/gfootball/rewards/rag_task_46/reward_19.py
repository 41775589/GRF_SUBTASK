import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to incentivize learning effective standing tackles and possession regaining."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.1  # Reward for tackling correctly
        self.penalty = -0.1  # Penalty for wrong tackle risking a foul

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "penalty": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            components["base_score_reward"][i] = reward[i]

            player_has_ball = (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'])
            opponent_has_ball = (o['ball_owned_team'] == 1)

            if opponent_has_ball:
                # Tackle reward scenario
                if 'action_sprint' in o['sticky_actions']:  # Simplified assumption of tackling
                    distance_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']][:2])
                    if distance_to_ball < 0.05:  # Close enough to realistically attempt a tackle
                        components["tackle_reward"][i] += self.tackle_reward
                        reward[i] += components["tackle_reward"][i]
                    else:
                        components["penalty"][i] += self.penalty
                        reward[i] += components["penalty"][i]

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
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
