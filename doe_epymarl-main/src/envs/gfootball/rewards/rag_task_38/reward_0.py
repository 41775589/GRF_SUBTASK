import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for long passes and quick transitions from defense to attack in football games."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.long_pass_reward = 0.5
        self.transition_reward = 0.3
        self.previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        """Reset the environment and clear the previous ball position tracking."""
        self.previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Add the state of the previous ball position to the pickle."""
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from the pickle, including the previous ball position."""
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle
    
    def reward(self, reward):
        """Adjust the reward based on effective long passes and quick transitions."""
        observation = self.env.unwrapped.observation()
        new_reward = [r for r in reward]
        components = {"base_score_reward": new_reward.copy(), "long_pass_reward": [0.0] * len(reward), "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            ball_position = o['ball'][:2]  # we consider only x and y coordinates
            if self.previous_ball_position is not None and o['ball_owned_team'] == 1:  # right team owns the ball
                distance_moved = np.linalg.norm(np.array(ball_position) - np.array(self.previous_ball_position))
                if distance_moved > 0.5:  # consider it a long pass if ball moved more than 0.5 units
                    new_reward[i] += self.long_pass_reward
                    components["long_pass_reward"][i] = self.long_pass_reward

                # Check for quick transition, observed by significant position change after regaining possession
                if 'ball_owned_team' in components and components['ball_owned_team'] != -1:  # change in possession
                    new_reward[i] += self.transition_reward
                    components["transition_reward"][i] = self.transition_reward

            self.previous_ball_position = ball_position  # update ball position for next calculation

        return new_reward, components

    def step(self, action):
        """Process the environment step, adjust rewards and include them in info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        # Also include data on sticky actions for each step.
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
        info.update({f"sticky_actions_{i}": self.sticky_actions_counter[i] for i in range(10)})
        
        return observation, reward, done, info
