import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering sliding tackles during specific defensive scenarios."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for idx, o in enumerate(observation):
            # Initialize component for this agent in this iteration.
            defensive_tackle_reward = 0.0

            # Identify defensive third conditions: our team must be close to our goal
            # and the ball must be controlled by the opponent.
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 1 else o['right_team'][o['active']]
            ball_pos = o['ball'][:2]  # Ignore z coordinate
            player_defensive_third = -0.7 < player_pos[0] < -0.3

            controlled_by_opponent_nearby = o['ball_owned_team'] not in [-1, o['ball_owned_team']]

            # Check if a slide action is active and effective
            sliding_action_active = o['sticky_actions'][9] == 1  # Assuming index 9 corresponds to slide action
            effective_tackle = np.linalg.norm(ball_pos - player_pos) < 0.05  # Effective if close to the ball

            # Reward for a successful sliding action in the defensive third during opponent control
            if player_defensive_third and controlled_by_opponent_nearby and sliding_action_active and effective_tackle:
                defensive_tackle_reward = 1.0

            # Accumulate specialized reward to the generic game reward
            reward[idx] += defensive_tackle_reward
            components[f"defensive_tackle_reward_{idx}"] = defensive_tackle_reward

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
