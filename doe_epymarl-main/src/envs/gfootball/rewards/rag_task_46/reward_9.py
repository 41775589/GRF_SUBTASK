import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward signals for mastering standing tackles and possession control."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle['previous_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward, "tackle_reward": 0}

        reward_components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0, 0.0]
        }

        for i, obs in enumerate(observation):
            # Tackling success is determined by regaining ball possession from opponent without fouls
            current_ball_owner_team = obs['ball_owned_team']
            if self.previous_ball_owner is not None and self.previous_ball_owner == 1 - current_ball_owner_team:
                if current_ball_owner_team != -1 and obs['game_mode'] == 0:  # normal play
                    reward_components["tackle_reward"][i] = 0.5  # reward for successful tackle and possession regain

            # Update the previous owner of the ball for the next reward calculation
            self.previous_ball_owner = current_ball_owner_team

            # Aggregate rewards
            reward[i] += reward_components["tackle_reward"][i]

        return reward, reward_components

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
