import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward based on effective collaboration between shooters and passers for scoring opportunities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooters_and_passers_rewards = {}
        self.shooting_boost = 0.1  # Extra reward for shots assisted by a recent pass.
        self.pass_boost = 0.05  # Extra reward for a pass leading to a scoring opportunity.
        self.shooters_and_passers_rewards = {}  # Track of each player's contribution towards collaborative plays.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.shooters_and_passers_rewards = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_rewards'] = self.shooters_and_passers_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooters_and_passers_rewards = from_pickle['checkpoint_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_boost": [0.0] * len(reward),
                      "shooting_boost": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            if o['ball_owned_team'] in [0, 1]:  # Check if the ball is owned by any team
                ball_owner = o['ball_owned_player']
                
                # Check if a successful pass happened recently
                if self.shooters_and_passers_rewards.get(ball_owner, {}).get('passed', False):
                    components["passing_boost"][idx] = self.pass_boost
                    self.shooters_and_passers_rewards[ball_owner]['passed'] = False  # Resetting the pass flag after reward

                # Check if the player is taking a shot possibly leading to a goal
                if any(action in [self.env.unwrapped.action_set.index('shot'), 
                                  self.env.unwrapped.action_set.index('long_shot')] 
                       for action in self.sticky_actions_counter):
                    components["shooting_boost"][idx] = self.shooting_boost
                    self.shooters_and_passers_rewards[ball_owner] = {'passed': False}

                # Update the base reward with additional collaborative play boosts
                reward[idx] += components["passing_boost"][idx] + components["shooting_boost"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
