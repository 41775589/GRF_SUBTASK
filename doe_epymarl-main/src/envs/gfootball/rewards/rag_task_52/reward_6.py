import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that improves defending strategies with specialized training for tackling, efficient movement, and passing under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackling_reward": np.zeros(len(reward)),
                      "positioning_reward": np.zeros(len(reward)),
                      "pressured_pass_reward": np.zeros(len(reward))}

        # Calculate additional rewards based on defensive actions
        for i, o in enumerate(observation):
            if 'game_mode' in o and 'ball_owned_team' in o:
                current_mode = o['game_mode']
                ball_owner = o['ball_owned_team']

                # Reward for successful tackle: when the ball ownership changes against the direction of attack
                if current_mode in [2, 3, 4, 5, 6] and ball_owner == 0:  # Assuming our team is the left team (team 0)
                    components['tackling_reward'][i] += 0.2

                # Reward for good positioning: when no goal is conceded in defensive modes
                if current_mode in [3, 4, 5] and ball_owner == 0:
                    components['positioning_reward'][i] += 0.1

                # Reward for successful passes under pressure: when in a defensive position and successfully pass
                if current_mode in [3, 4, 5] and 'ball_direction' in o:
                    if np.linalg.norm(o['ball_direction'][:2]) > 0.1:  # Assuming significant ball movement
                        components['pressured_pass_reward'][i] += 0.15

                # Integrating components rewards into the primary reward
                total_additional_reward = (components['tackling_reward'][i] 
                                           + components['positioning_reward'][i]
                                           + components['pressured_pass_reward'][i])
                reward[i] += total_additional_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
