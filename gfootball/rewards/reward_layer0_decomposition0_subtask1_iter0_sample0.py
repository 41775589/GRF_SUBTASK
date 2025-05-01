import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive and midfield strategic performance."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pos_session_start = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pos_session_start = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.pos_session_start
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pos_session_start = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Establishing reward components for tracking
        components = {"base_score_reward": reward.copy(),
                      "defense_midfield_control": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            obs = observation[idx]

            # Check if defense and midfield players control the ball effectively at the start
            # Give reward for maintaining good ball control with defensive and midfield players
            if obs['ball_owned_team'] == 0:  # Assuming 0 is the agent's team
                player_pos = obs['left_team'][obs['ball_owned_player']]
                ball_pos = obs['ball']

                # Define midfield area as x ranges anywhere and y within [-0.14, 0.14]
                if abs(player_pos[1]) <= 0.14:
                    # Reward for midfield ball control
                    components['defense_midfield_control'][idx] += 0.05
                    reward[idx] += 0.05

                # Define defense area as x < -0.5 and y within [-0.42, 0.42]
                if player_pos[0] < -0.5:
                    # Reward for defensive positioning and ball control
                    components['defense_midfield_control'][idx] += 0.1
                    reward[idx] += 0.1

            # Encourage not losing the ball quickly from defense or midfield start
            if idx not in self.pos_session_start:
                self.pos_session_start[idx] = obs['ball_owned_team'] == 0 and obs['ball_owned_player'] != -1
            else:
                if self.pos_session_start[idx] and (obs['ball_owned_team'] != 0 or obs['ball_owned_player'] == -1):
                    reward[idx] -= 0.1
                    components['defense_midfield_control'][idx] -= 0.1
                    self.pos_session_start[idx] = False

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
