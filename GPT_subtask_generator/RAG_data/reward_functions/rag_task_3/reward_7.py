import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward function for shooting skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_close_attempts = 0
        self.shots_on_target = 0
        self.power_threshold = 0.5  # Hypothetical threshold for shot power measurement

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_close_attempts = 0
        self.shots_on_target = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shots_on_target'] = self.shots_on_target
        to_pickle['previous_close_attempts'] = self.previous_close_attempts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shots_on_target = from_pickle.get("shots_on_target", 0)
        self.previous_close_attempts = from_pickle.get("previous_close_attempts", 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for shot power and towards the goal
            if 'ball_direction' in o:
                ball_dir = o['ball_direction'][:2]
                goal_dir = np.array([1, 0])  # Assuming right goal is the target

                # Check if shot is towards the goal
                dot_product = np.dot(ball_dir, goal_dir)
                power_measured = np.linalg.norm(ball_dir)
                
                # Reward shots on target with sufficient power
                if dot_product > 0.85 and power_measured > self.power_threshold:
                    components["shooting_accuracy_bonus"][rew_index] = 1.0
                    reward[rew_index] += components["shooting_accuracy_bonus"][rew_index]
                    self.shots_on_target += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
