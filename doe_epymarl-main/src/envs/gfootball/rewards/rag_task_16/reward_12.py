import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes with precision in a soccer environment."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_threshold = 0.1  # Threshold for considering a pass as 'highly precise'
        self.high_pass_reward = 0.5  # Reward for executing a precise high pass

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the environment."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the environment."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the reward based on the precision of high passes performed by agents."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball = o.get('ball', [0, 0, 0])
            ball_owned_team = o.get('ball_owned_team', -1)
            ball_direction = o.get('ball_direction', [0, 0, 0])
            
            # Checking if the ball is moving in a 'high' trajectory and is owned by the team
            if ball[2] > self.pass_accuracy_threshold and ball_owned_team == 0:
                # Check the direction to ensure it is a forward pass
                if ball_direction[0] > self.pass_accuracy_threshold:
                    components["high_pass_reward"][rew_index] = self.high_pass_reward
                    reward[rew_index] += self.high_pass_reward

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
