import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective clearance of the ball from defensive zones under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.penalty_zone_threshold = -0.6
        self.clearance_threshold = 0.0
        self.clearance_success_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and internal states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the current internal state."""
        to_pickle['StickyActions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the internal state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['StickyActions']
        return from_pickle

    def reward(self, reward):
        """Calculate extra reward based on ball clearance from defensive zones under pressure."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clearance_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if ball is in defensive zone and clear it from there
            if o['ball'][0] < self.penalty_zone_threshold:
                if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                    # Calculate the distance the ball moves towards midfield or opposing half after a kick
                    distance_cleared = o['ball'][0] - self.penalty_zone_threshold
                    # Reward the agent for clearing the ball beyond the defensive penalty area
                    if distance_cleared > self.clearance_threshold:
                        components['clearance_reward'][rew_index] = self.clearance_success_reward
                        reward[rew_index] += components['clearance_reward'][rew_index]
                        
        return reward, components

    def step(self, action):
        """Execute a step in the environment, record and adjust rewards, and log information."""
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
