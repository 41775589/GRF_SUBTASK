import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances team synergy during possession changes,
    emphasizing precise timing and strategic positioning for effective offense and defense."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save sticky actions counter state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Retrieve sticky actions counter state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Add rewards for possession changes with strategic positioning and timing."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, copy=True),
                      "positional_change_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, (o, rew) in enumerate(zip(observation, reward)):
            # Reward for changing possession under correct conditions
            ball_owned_change = o['ball_owned_team'] != self.env.unwrapped.previous_ball_owned_team
            if ball_owned_change and o['game_mode'] == 0:  # Normal game mode
                if o['ball'][0] * np.sign(o['ball_direction'][0]) > 0.7:  # Ball is moving towards opponent's half
                    components['positional_change_reward'][rew_index] = 0.5
                    reward[rew_index] += components['positional_change_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Perform a step using the given action, with added reward components and final reward adjustments."""
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
