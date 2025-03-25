import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to enhance offensive skills through fast-paced maneuvers and precision-control.
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialization of counters and reward modifiers
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky action counters.
        """
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the environment for saving or serialization.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment for loading from serialization.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Compute a custom reward based on the positions and actions promoting fast-paced offensive play.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "fast_play_boost": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Encourage quick forward movements and passing in opponent's half
            if o['ball_owned_team'] == 1:  # Assuming the right team is the one enhanced
                ball_x_position = o['ball'][0]
                ball_speed_x = o['ball_direction'][0]

                # Bonus for having the ball in the opponent's half and moving forward
                if ball_x_position > 0 and ball_speed_x > 0:
                    components['fast_play_boost'][rew_index] = 0.1  # Some small positive reward

                # Additional bonus if near opponent's goal and controlling the ball
                if ball_x_position > 0.5:
                    components['fast_play_boost'][rew_index] += 0.2  # More significant reward for being close to scoring

            # Aggregate the reward components
            reward[rew_index] += components['fast_play_boost'][rew_index]

        return reward, components

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
