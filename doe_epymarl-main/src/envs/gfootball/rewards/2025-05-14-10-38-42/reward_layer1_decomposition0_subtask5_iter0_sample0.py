import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering defensive passes under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encodes the current state of the wrapper along with the environment state."""
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Decodes and sets the state of the environment and the wrapper from a pickle."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Custom reward function focusing on passes from defensive positions."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}

        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": 0.0  # initialize with no additional reward
        }

        o = observation[0]  # considering a single-agent scenario
        ball_owned_team = o['ball_owned_team']

        # Reward for successful passing when in defensive positions
        if ball_owned_team == 0:  # If the ball is owned by the agent's team
            player_x, player_y = o['left_team'][o['active']]
            if player_x < -0.5:  # Defensive half
                component_reward = 1.0
                components['passing_reward'] = component_reward
                reward += component_reward

        return reward, components

    def step(self, action):
        """Steps through the environment, returning wrapped observations and rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        info.update({f"component_{key}": sum(value) for key, value in components.items()})
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
