import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that aims to train agents in maintaining ball control under pressure and making strategic plays.
    This is done by giving rewards for passing the ball effectively across different player-defined checkpoints
    and maintaining control when close to these checkpoints."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.checkpoints = [np.array([-0.5, 0.5]), np.array([0.0, 0.0]), np.array([0.5, -0.5])]
        self.checkpoint_rewards = np.full(len(self.checkpoints), 0.05)
        self.checkpoint_collected = [False]*len(self.checkpoints)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and checkpoint states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoint_collected = [False]*len(self.checkpoints)
        return self.env.reset()

    def reward(self, reward):
        """Customize reward based on checkpoint proximity and ball handling under pressure."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_rewards": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            ball_position = o['ball'][:2]  # get [x, y] of the ball

            # Reward for proximity to each checkpoint
            for idx, checkpoint in enumerate(self.checkpoints):
                if self.checkpoint_collected[idx]:
                    continue
                distance = np.linalg.norm(ball_position - checkpoint)
                if distance < 0.1:  # If within a small radius of the checkpoint
                    components["checkpoint_rewards"][i] += self.checkpoint_rewards[idx]
                    self.checkpoint_collected[idx] = True

            # Update total reward value
            reward[i] += components["checkpoint_rewards"][i]
        
        return reward, components

    def step(self, action):
        """Override step to include customized rewards and possibly other environmental components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        info['checkpoint_status'] = self.checkpoint_collected
        return observation, reward, done, info
