import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on defensive actions by a stopper."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_positions_at_ball_loss = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_positions_at_ball_loss = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'player_positions_at_ball_loss': self.player_positions_at_ball_loss
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.player_positions_at_ball_loss = from_pickle['CheckpointRewardWrapper']['player_positions_at_ball_loss']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_actions_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:
                # Reward for aggressive tight marking when the opposing team owns the ball
                player_pos = o['right_team'][o['active']]
                ball_pos = o['ball'][:2]
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)
                if distance_to_ball < 0.05:  # very tight marking
                    components['defensive_actions_reward'][rew_index] += 0.1
                    reward[rew_index] += components['defensive_actions_reward'][rew_index]

            if o['game_mode'] in [2, 3, 4]:  # It's a defensive game mode (GoalKick, FreeKick, Corner)
                components['defensive_actions_reward'][rew_index] += 0.2
                reward[rew_index] += components['defensive_actions_reward'][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
