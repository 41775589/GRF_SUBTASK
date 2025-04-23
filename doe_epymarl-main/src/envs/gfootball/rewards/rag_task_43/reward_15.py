import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive strategies and transitions into counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_positioning": [0.0] * len(reward),
            "quick_transition": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' not in o:
                continue

            # Reward for maintaining good positions defensively
            if o['ball_owned_team'] == 1:  # Opponent has the ball
                # Calculate distance of all players to the ball
                distances = np.linalg.norm(o['left_team'] - o['ball'][:2], axis=1)
                # Reward for players being within 0.3 distance units from the ball
                components["defensive_positioning"][rew_index] = np.sum(distances < 0.3) * 0.05

            # Reward for quick transition to counterattack
            if o['ball_owned_team'] == 0:  # Own team has the ball
                prior_pos = o['left_team'][o['ball_owned_player']]
                player_speed = np.linalg.norm(o['left_team_direction'][o['ball_owned_player']])
                # Reward for advancing forward past mid-field (x > 0) with the ball at speed > 0.01
                if prior_pos[0] > 0 and player_speed > 0.01:
                    components["quick_transition"][rew_index] = 0.1

            # Combine the rewards
            reward[rew_index] += components["defensive_positioning"][rew_index]
            reward[rew_index] += components["quick_transition"][rew_index]
        
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
