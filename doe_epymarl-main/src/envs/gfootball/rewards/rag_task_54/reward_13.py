import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom reward wrapper to encourage collaborative plays between shooters and passers.
    This is designed to enhance team play for exploiting scoring opportunities fully.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "cooperative_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Reward for successful passing (ball_owned_player changes and is near a teammate)
            if o['ball_owned_team'] == 0 and 'ball_owned_player' in o:
                current_owner = o['ball_owned_player']
                ball_pos = o['ball'][:2]  # X, Y position

                for j, pos in enumerate(o['left_team']):
                    if j != current_owner:
                        dist = np.linalg.norm(pos - ball_pos)
                        if dist < 0.05:  # Threshold to consider a player close
                            components['passing_reward'][i] += 0.1

            # If a pass results in a shot (from passer to shooter) that progresses towards goal (increase in x)
            # This pushes teamwork focused on moving the ball forward towards opponent's goal.
            if o['right_team_active'].sum() > 0:  # At least one player is active in the opposing team
                active_players = np.where(o['left_team_active'])[0]

                for player_index in active_players:
                    if 'ball_owned_player' in o and player_index == o['ball_owned_player']:
                        if o['ball'][0] > 0.5:  # Ball is moved closer to the opponent's goal (X > 0.5)
                            components['cooperative_play_reward'][i] += 0.2

        # Aggregate the rewards and components
        for idx, base_rew in enumerate(reward):
            reward[idx] += components['passing_reward'][idx] + components['cooperative_play_reward'][idx]

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
