import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focusing on the defensive role of 'stopper'."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.2
        self.block_shot_reward = 0.3
        self.stall_reward = 0.1
        self.last_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['interceptions_count'] = self.interception_reward
        to_pickle['block_shot_count'] = self.block_shot_reward
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interception_reward = from_pickle['interceptions_count']
        self.block_shot_reward = from_pickle['block_shot_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "block_shot_reward": [0.0] * len(reward),
            "stall_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Defensive rewards for 'stopper' role
            ball_owner_now = o['ball_owned_team'] == 1  # Assuming the agent team is 1
            if self.last_ball_owner is not None and self.last_ball_owner and not ball_owner_now:
                components["interception_reward"][rew_index] = self.interception_reward
                reward[rew_index] += components["interception_reward"][rew_index]
            
            if o['game_mode'] in [3, 5]:  # Freekick or Throw-in for the opponent
                components["block_shot_reward"][rew_index] = self.block_shot_reward
                reward[rew_index] += components["block_shot_reward"][rew_index]
            
            # Use sticky actions to check if the player is actively blocking opponents (sprint, dribble)
            active_defense = (o['sticky_actions'][8] or o['sticky_actions'][9])  # Sprint or Dribble
            if active_defense and ball_owner_now:
                components["stall_reward"][rew_index] = self.stall_reward
                reward[rew_index] += components["stall_reward"][rew_index]

            # Update last ball owner status
            self.last_ball_owner = ball_owner_now

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
