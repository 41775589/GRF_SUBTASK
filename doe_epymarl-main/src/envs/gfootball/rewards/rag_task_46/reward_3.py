import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on successful standing tackles and ball possession gain without fouls."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.yellow_card_state = None
        self.ball_ownership_changes = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.yellow_card_state = None
        self.ball_ownership_changes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['yellow_card_state'] = self.yellow_card_state
        to_pickle['ball_ownership_changes'] = self.ball_ownership_changes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.yellow_card_state = from_pickle['yellow_card_state']
        self.ball_ownership_changes = from_pickle['ball_ownership_changes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": 0.0}

        if observation is None:
            return reward, components

        game_mode = observation[0]['game_mode']
        ball_owned_team = observation[0]['ball_owned_team']
        prev_ball_owned_team = observation[1]['ball_owned_team'] if self.yellow_card_state is not None else None

        # Reward players for changing ball possession through tackles, not fouls.
        if (0 in observation[0]['left_team_yellow_card'] or 1 in observation[0]['right_team_yellow_card'])\
                and self.yellow_card_state != observation[0]['left_team_yellow_card']:
            components["tackle_reward"] -= 0.1  # Penalize for gaining possession via foul

        if ball_owned_team != prev_ball_owned_team and prev_ball_owned_team is not None:
            components["tackle_reward"] += 0.5  # Reward for changing possession legally
            self.ball_ownership_changes += 1

        reward += components["tackle_reward"]
        
        self.yellow_card_state = observation[0]['left_team_yellow_card']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        info.update({f"component_{key}": value for key, value in components.items()})
        return observation, reward, done, info
