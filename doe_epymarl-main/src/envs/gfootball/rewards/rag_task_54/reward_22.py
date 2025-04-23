import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on collaborative plays between shooters and passers."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_checkpoints = {}
        self.shooting_checkpoints = {}
        self.reward_for_passing = 0.05
        self.reward_for_shooting = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_checkpoints = {}
        self.shooting_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "passing_checkpoints": self.passing_checkpoints,
            "shooting_checkpoints": self.shooting_checkpoints
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_from_pickle = from_pickle['CheckpointRewardWrapper']
        self.passing_checkpoints = state_from_pickle["passing_checkpoints"]
        self.shooting_checkpoints = state_from_pickle["shooting_checkpoints"]
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for i, (rew, obs) in enumerate(zip(reward, observation)):
            ball_owner_team = obs['ball_owned_team']
            if ball_owner_team != -1: # Either team owns the ball
                # Checking for passes (assuming 8 is dribble action, 9 is sprint)
                if obs['sticky_actions'][8] == 1 and obs['ball_owned_player'] != -1:
                    # Action represents passing if dribble is active
                    if i not in self.passing_checkpoints:
                        components["passing_reward"][i] = self.reward_for_passing
                        reward[i] += components["passing_reward"][i]
                        self.passing_checkpoints[i] = True

                # Checking for shots (assuming 5 is a potential shooting action right direct towards goal)
                if obs['sticky_actions'][5] == 1 and obs['ball_owned_player'] == obs['designated']:
                    # Action represents shooting towards goal if bottom_right is active
                    if i not in self.shooting_checkpoints:
                        components["shooting_reward"][i] = self.reward_for_shooting
                        reward[i] += components["shooting_reward"][i]
                        self.shooting_checkpoints[i] = True

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        return obs, reward, done, info
