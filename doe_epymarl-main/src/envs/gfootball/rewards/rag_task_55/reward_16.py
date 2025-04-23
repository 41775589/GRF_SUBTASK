import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on defensive tactics involving sliding and standing tackles without fouling."""

    def __init__(self, env):
        super().__init__(env)
        self.ball_ownership_changes = 0
        self.total_tackles = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.fouls_committed = 0
        self.rewards_for_tackles = 0.5
        self.penalty_for_fouls = -0.3

    def reset(self):
        """Resets the environment's variables."""
        self.ball_ownership_changes = 0
        self.total_tackles = 0
        self.fouls_committed = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Compute additional defensive reward components based on tackles and fouls."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            if obs['game_mode'] in [3, 4, 6]:  # FreeKick, Corner, Penalty (indicative of fouls)
                self.fouls_committed += 1
                components["foul_penalty"][rew_index] = self.penalty_for_fouls
                reward[rew_index] += components["foul_penalty"][rew_index]
            
            if obs['sticky_actions'][7] or obs['sticky_actions'][8]:  # Tackle or Slide actions
                self.total_tackles += 1
                components["tackle_reward"][rew_index] = self.rewards_for_tackles
                reward[rew_index] += components["tackle_reward"][rew_index]
            
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
