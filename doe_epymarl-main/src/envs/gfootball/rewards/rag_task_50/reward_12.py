import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that promotes the agent for executing long, accurate passes across specific checkpoints."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_checkpoints = np.array([
            [-0.75, 0],  # left midfield
            [0, -0.4],   # center back
            [0, 0.4],    # center forward
            [0.75, 0]    # right midfield
        ])
        self.checkpoint_rewards = np.zeros(4)
        self.old_ball_pos = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoint_rewards.fill(0)
        self.old_ball_pos = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "checkpoint_rewards": self.checkpoint_rewards,
            "old_ball_pos": self.old_ball_pos
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoint_rewards = from_pickle['CheckpointRewardWrapper']['checkpoint_rewards']
        self.old_ball_pos = from_pickle['CheckpointRewardWrapper']['old_ball_pos']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward),
                      "passing_reward": np.zeros(len(reward))}
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and self.old_ball_pos is not None:
                new_ball_pos = o['ball'][:2]  # consider only x, y position of the ball
                ball_travel = np.linalg.norm(new_ball_pos - self.old_ball_pos)

                # Reward long passes that travel through specific zones
                if ball_travel > 0.5:  # Consider it a 'long' pass
                    for idx, cp in enumerate(self.pass_checkpoints):
                        if np.linalg.norm(new_ball_pos - cp) < 0.1:  # Close to checkpoint
                            if self.checkpoint_rewards[idx] == 0:
                                components["passing_reward"][i] += 0.1
                                self.checkpoint_rewards[idx] = 1  # Reward only once per episode
                                break  # Reward for the first checkpoint hit

            self.old_ball_pos = o['ball'][:2]

        new_rewards = components["base_score_reward"] + components["passing_reward"]
        return new_rewards.tolist(), components

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
                self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
