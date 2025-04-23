import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized midfield dynamics reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_checkpoint_reward = 0.05
        self.controlled_checkpoints = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.controlled_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['MidfieldCheckpointRewardWrapper'] = self.controlled_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.controlled_checkpoints = from_pickle['MidfieldCheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward for central midfielders advancing the ball softly upfield.
            if ('left_team_roles' in o and o['active'] in o['left_team_roles'] and 
                    o['left_team_roles'][o['active']] in [5, 6] and   # Midfield Roles
                    'ball_owned_team' in o and o['ball_owned_team'] == 0 and
                    'ball_owned_player' in o and o['ball_owned_player'] == o['active']):

                # Calculate approximate midfield sector (y-axis)
                if abs(o['ball'][1]) <= 0.15:
                    key = f"midfield_{o['active']}"
                    if key not in self.controlled_checkpoints:
                        self.controlled_checkpoints[key] = True
                        reward[i] += self.midfield_checkpoint_reward
                        components["midfield_control_reward"][i] = self.midfield_checkpoint_reward
        
            # Additionally manage rewards for other player interactions.
            # This could include penalizing losses of ball control or
            # rewarding defensive actions, depending on game dynamics.

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
