import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to improve dribbling skills against the goalkeeper with emphasis on feints and sudden changes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_has_ball = False
        self.prev_ball_pos = (0, 0)
        self.dribble_success_reward = 0.5
        self.feint_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_has_ball = False
        self.prev_ball_pos = (0, 0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = dict(
            sticky_actions_counter=self.sticky_actions_counter.tolist(),
            player_has_ball=self.player_has_ball,
            prev_ball_pos=self.prev_ball_pos)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_dict = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = np.array(state_dict['sticky_actions_counter'], dtype=int)
        self.player_has_ball = state_dict['player_has_ball']
        self.prev_ball_pos = tuple(state_dict['prev_ball_pos'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_success_reward": [0.0] * len(reward),
                      "feint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                # Player has the ball
                if not self.player_has_ball:
                    self.player_has_ball = True
                current_pos = o['ball'][:2]
                
                # Check for feints or sudden movements when owning the ball
                if self.prev_ball_pos != (0, 0):
                    distance_moved = np.linalg.norm(np.array(self.prev_ball_pos) - np.array(current_pos))
                    if distance_moved > 0.1:  # Assuming a significant sudden movement
                        components['feint_reward'][i] = self.feint_reward
                
                reward[i] += components['feint_reward'][i]
                self.prev_ball_pos = current_pos
            else:
                self.player_has_ball = False
                self.prev_ball_pos = (0, 0)

            # For dribbling reward based on maintaining ball control
            if o['sticky_actions'][9] == 1:  # Dribbling action is active
                components['dribble_success_reward'][i] = self.dribble_success_reward
                reward[i] += components['dribble_success_reward'][i]

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
        return observation, reward, done, info
