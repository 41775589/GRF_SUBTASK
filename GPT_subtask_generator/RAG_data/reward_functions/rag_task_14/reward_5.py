import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for the sweeper role in a football game."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_threshold = -0.25  # Defensive zone boundary threshold
        self.clearance_reward = 0.3       # Reward for clearing the ball from the defensive zone
        self.tackle_reward = 0.2          # Reward for tackling in the defensive zone

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clearance_reward": [0.0] * len(reward),
            "tackle_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos_x = o['ball'][0]
            player_pos_x = o['left_team'][o['active']][0]  # active player's x position

            # Reward for clearing the ball from the defensive zone
            if ball_pos_x < self.clearance_threshold < player_pos_x:
                if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # ball is owned by the active player's team
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += components["clearance_reward"][rew_index]

            # Additional tackle reward in defensive zone
            if player_pos_x < self.clearance_threshold:
                # In defensive zone, check if action is a tackle and successful (could use specific game mode flags)
                if o['game_mode'] == 5 and 'action' in o and o['action'] == football_action_set.CoreAction.action_sliding:
                    components["tackle_reward"][rew_index] = self.tackle_reward
                    reward[rew_index] += components["tackle_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Track sticky actions for possible analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
