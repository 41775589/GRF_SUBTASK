import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to augment reward by focusing on maintaining strategic positioning
    using all directional movements, ensuring effective pivoting between defensive
    stance and initiating attacks.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        strategic_positioning_reward = np.zeros(len(reward))
        for idx, obs in enumerate(observation):
            ball_position = obs['ball'][:2]  # Only use x, y coordinates of the ball
            player_position = obs['left_team'][obs['active']][:2] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']][:2]

            # Reward for being close to the ball when in defensive position or when ready to initiate an attack
            distance_to_ball = np.linalg.norm(ball_position - player_position)
            if distance_to_ball < 0.1:  # threshold for closeness
                strategic_positioning_reward[idx] += 0.05
            
            # Reward for correct pivoting: utilizing all movements efficiently
            # Count all active sticky actions which suggest movement (ignoring other actions like sprint/dribble)
            active_movements = np.sum(obs['sticky_actions'][:8])
            max_possible_actions = 8  # Only consider directional stick actions
            efficiency_ratio = active_movements / max_possible_actions
            strategic_positioning_reward[idx] += 0.02 * efficiency_ratio
        
        # Update components dictionary to track individual reward components
        components.update({"strategic_positioning_reward": strategic_positioning_reward.tolist()})

        # Apply the calculated extra rewards to the original rewards
        reward += strategic_positioning_reward
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
