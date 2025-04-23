import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a sliding tackle mastery reward, emphasizing timing and player positioning during counter-attacks and defensive phases."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
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
        components = {"base_score_reward": reward.copy(), 
                      "tackle_timing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['game_mode'] in [3, 6] and o['ball_owned_team'] == 1:
                # Calculate the distance to the ball and opponent's direction
                distance_to_ball = np.linalg.norm(np.array(o['ball'][:2]) - np.array(o['left_team'][o['designated']]))
                opponent_direction = o['right_team_direction'][o['designated']]

                # Check if the ball is moving towards the player and the proximity triggers a potential tackle
                if np.dot(opponent_direction, o['ball_direction'][:2]) < 0 and distance_to_ball <= 0.1:
                    components["tackle_timing_reward"][rew_index] = 0.5  # Reward for well-timed sliding tackle

            # Integrate new component reward
            reward[rew_index] += components["tackle_timing_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
