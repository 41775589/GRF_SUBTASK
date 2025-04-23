import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward based on shooting angles and timing under pressure near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        components = {"base_score_reward": reward.copy()}
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        # Assuming there are two agents with their respective observations in the list.
        for idx in range(len(reward)):
            o = observation[idx]

            # Check if the active player is near the opponent's goal under pressure and has the ball.
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                # Calculate distance to opponent goal based on x-coordinate of the field (Y-axis)
                distance_to_goal = 1 - abs(o['ball'][0])
                angle_factor = abs(o['ball'][1])  # Y-coordinate judgment for determining the shooting angle

                # Increase reward significantly if close to goal with a direct angle and under game pressure (mode != 0)
                if distance_to_goal < 0.3 and angle_factor < 0.1 and o['game_mode'] != 0:
                    bonus = 1.0  # Large bonus for difficult goal situations
                    reward[idx] += bonus
                    components.setdefault('pressure_goal_bonus', []).append(bonus)
                else:
                    components.setdefault('pressure_goal_bonus', []).append(0.0)
            else:
                components.setdefault('pressure_goal_bonus', []).append(0.0)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        updated_reward, components = self.reward(reward)
        info.update({f'component_{k}': sum(v) for k, v in components.items()})
        info['final_reward'] = sum(updated_reward)
        info['base_score_reward'] = sum(components['base_score_reward'])

        return observation, updated_reward, done, info
