import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive skill reward focusing on interception, marking, tackling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky actions counter and other necessary states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the current state of the environment along with the wrapper's state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the state of the environment along with the wrapper's state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Compute the reward based on the agent's defensive actions."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}

        for idx in range(len(reward)):
            o = observation[idx]
            if o['ball_owned_team'] != 1:  # Focus when opponent has ball possession
                continue
            
            # Encourage defensive actions by the player
            sticky_actions = o['sticky_actions']
            action_sliding, action_stop_dribble, action_stop_moving = sticky_actions[6], sticky_actions[9], sticky_actions[0]
            action_count = action_sliding + action_stop_dribble + action_stop_moving
            defensive_bonus = 0.1 * action_count

            components["defensive_reward"][idx] = defensive_bonus
            reward[idx] += defensive_bonus
        
        return reward, components

    def step(self, action):
        """Process the environment step and augment reward with defense specific metrics."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
