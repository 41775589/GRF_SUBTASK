import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes clearing the ball from defensive zones efficiently under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_threshold = 0.3  # Midfield line in normalized coordinates
        self.clearance_reward = 1.0
        self.defensive_zone_start = -1.0  # Own goal line
        self.defensive_zone_end = -0.2  # Deep into defensive half
        
    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the environment."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        """Set the state of the environment."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Calculate the reward for the agent's actions, especially focused on clearances."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "clearance_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['game_mode'] == 0:  # Normal gameplay
                ball_x = o['ball'][0]

                # Check ball position in the defensive zone and if the ball moves towards midfield or further
                if self.defensive_zone_start <= ball_x <= self.defensive_zone_end:
                    if 'ball_direction' in o and abs(o['ball_direction'][0]) > 0:
                        # The ball is moving towards the midfield or opponent's side
                        movement_direction = np.sign(o['ball_direction'][0])
                        if movement_direction > 0:
                            components["clearance_reward"][rew_index] = self.clearance_reward
                            reward[rew_index] += self.clearance_reward
                        
        return reward, components

    def step(self, action):
        """Implement the environment's step function, gather data, and apply the new reward function."""
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
