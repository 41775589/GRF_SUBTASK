import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for performing offensive skills like passing,
    shooting, dribbling, and quick movement.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Weight factors for various actions
        self.pass_reward = 0.1
        self.shoot_reward = 0.3
        self.dribble_reward = 0.05
        self.sprint_reward = 0.02

        # Track cumulative rewards for specific actions
        self.action_rewards = np.zeros((env.action_space.n,), dtype=float)

        # Initialize action mapping indices
        self.action_map = {
            'short_pass': 0, 
            'long_pass': 1, 
            'high_pass': 2, 
            'shot': 3, 
            'dribble': 8,
            'sprint': 9
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        # Initialize modified reward and a component dictionary for tracking the-detailed rewards
        modified_reward = np.array(reward, copy=True)
        components = {"base_score_reward": reward.copy(),
                      "action_rewards": np.zeros_like(reward)}

        for i, obs in enumerate(observation):
            active_player_actions = obs['sticky_actions'][self.action_map['sprint']:self.action_map['dribble']+1]
            components['action_rewards'][i] += self.sprint_reward * active_player_actions[self.action_map['sprint']]
            components['action_rewards'][i] += self.dribble_reward * active_player_actions[self.action_map['dribble']]
            
            if obs['game_mode'] == 0:  # Normal play mode
                if obs['ball_owned_player'] == obs['active']:
                    # Player has the ball and can take actions such as pass or shoot
                    pass_actions = obs['sticky_actions'][self.action_map['short_pass']:self.action_map['high_pass']+1]
                    shot = obs['sticky_actions'][self.action_map['shot']]

                    components['action_rewards'][i] += self.pass_reward * np.sum(pass_actions)
                    components['action_rewards'][i] += self.shoot_reward * shot

            # Aggregate base and extra rewards
            modified_reward[i] += components['action_rewards'][i]

        return modified_reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Add reward components to info for monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # monitor sticky actions usage
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active

        return obs, reward, done, info
