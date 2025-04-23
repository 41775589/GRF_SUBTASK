import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to improve mastering midfield dynamics including coordination and strategic transitions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
        self.num_zones = 5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.position_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.position_rewards = from_picle['CheckpointRewardWrapper']
        return from_picle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "position_reward": [0.0] * len(reward)
        }

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            if 'game_mode' in obs and obs['game_mode'] != 0:
                # No rewards for game transitions like goals, kick-offs, etc.
                continue

            mid_pos_x = 0
            if not obs['ball_owned_player'] == obs['active']:
                # We only interest in the situation when the controlled player also owns the ball
                continue

            player_pos = obs['right_team'][obs['active']]

            # Simple midfield delimiter, horizontally near zero in X-coordinate
            if player_pos[0] > mid_pos_x:
                distance_to_mid = np.abs(player_pos[0] - mid_pos_x)
                reward_component = np.exp(-distance_to_mid)
                components['position_reward'][idx] += reward_component
                reward[idx] += reward_component

        # Update the reward based on the components before returning
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Append the modified reward and component breakdown into info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Capture sticky actions for additional analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
