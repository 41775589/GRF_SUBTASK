import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for practicing offensive strategies in football."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, rewards):
        # Initialize the components of the rewards for each player.
        components = {
            "base_score_reward": [reward.copy() for reward in rewards],
            "shooting_reward": [0.0] * len(rewards),
            "dribbling_reward": [0.0] * len(rewards),
            "passing_reward": [0.0] * len(rewards)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return rewards, components

        assert len(rewards) == len(observation)

        for idx, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # If right team controls the ball.
                if o['ball_owned_player'] == o['active']:  # If ball is owned by the active player.
                    # Reward for shooting when near the goal.
                    if np.linalg.norm(o['ball'][:2] - np.array([1, 0])) < 0.2:
                        components["shooting_reward"][idx] = 0.3

                    # Reward for effective dribbling (if dribble action is active).
                    if o['sticky_actions'][9] == 1:  # Action dribble index.
                        components["dribbling_reward"][idx] = 0.1

                    # Reward for successful passes (based on change of ball ownership).
                    previous_ball_owner = self.env.unwrapped.previous_ball_owner
                    if previous_ball_owner is not None and previous_ball_owner != o['active']:
                        components["passing_reward"][idx] = 0.2

            # Summing up the rewards for this player.
            rewards[idx] = (1 * components['base_score_reward'][idx] +
                            components['shooting_reward'][idx] +
                            components['dribbling_reward'][idx] +
                            components['passing_reward'][idx])

        return rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Adding reward components to info for analysis.
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
