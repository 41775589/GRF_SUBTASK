import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense teamwork-focused reward, particularly on plays involving passing and shooting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05     # Reward increment for effective passes
        self.shot_reward = 0.1      # Reward increment for effective shots
        self.goals_scored = 0       # Keeping track of goals scored

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goals_scored = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['goals_scored'] = self.goals_scored
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goals_scored = from_pickle['goals_scored']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, obs in enumerate(observation):
            # Reward for passes resulting in a shot
            if obs['game_mode'] in {2, 3, 4, 5, 6}:
                for i, action in enumerate(obs['sticky_actions'][-2:]):  # focusing on dribble and sprint actions as proxies for progressive passes
                    if action > 0 and obs['ball_owned_team'] == 0:  # Team 0 owns the ball
                        components['pass_reward'][rew_index] += self.pass_reward
                        components['shot_reward'][rew_index] += self.shot_reward
                        reward[rew_index] += components['pass_reward'][rew_index] + components['shot_reward'][rew_index]

            # Additional reward for teamwork leading to a goal
            current_score = obs['score'][0]   # Assuming index 0 is the monitored team's score
            if current_score > self.goals_scored:
                self.goals_scored = current_score   # Update goals tracked
                reward[rew_index] += 1.0  # Equivalent to scoring a goal

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
