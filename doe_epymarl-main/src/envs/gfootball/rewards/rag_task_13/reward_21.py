import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for 'stopping' actions in defense.

    This reward wrapper focuses on the role of a 'stopper', which includes enhancing skills in intense man-marking, blocking shots, and stalling forward moves by opposing players.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_rewards": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for idx, player_obs in enumerate(observation):
            blocking_reward = 0
            # Reward the player for effective defensive positions (e.g., close to an opposing player with the ball)
            if player_obs['ball_owned_team'] == 1:  # if the ball is with the opposite team
                distance_to_ball = np.linalg.norm(player_obs['ball'][:2] - player_obs['right_team'][player_obs['active']][:2])
                if distance_to_ball < 0.1:  # in close proximity to the ball
                    blocking_reward += 0.1  # small reward for staying close to ball

                # if the 'active' player just blocked a potential shot or forward pass
                if player_obs['game_mode'] in [3, 4, 6]:  # modes that likely follow a block action (FreeKick, Corner, Penalty)
                    blocking_reward += 0.2

            # Update the reward for this player with defensive components
            components["defensive_rewards"][idx] = blocking_reward
            reward[idx] += blocking_reward
        
        return reward, components

    def step(self, action):
        observations, rewards, done, info = self.env.step(action)
        new_rewards, components = self.reward(rewards)
        info["final_reward"] = sum(new_rewards)
        info.update({f"component_{k}": sum(v) for k, v in components.items()})

        return observations, new_rewards, done, info
