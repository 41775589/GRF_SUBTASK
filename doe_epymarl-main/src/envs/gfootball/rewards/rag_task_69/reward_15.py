import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a structured reward focused on offensive strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shoot_distance_threshold = 0.2  # Approximate shooting range/window
        self.pass_accuracy_threshold = 0.1   # Accuracy requirement for passes
        self.dribble_effectiveness_threshold = 0.1  # How much dribbling near opponents counts
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]

            # Shooting near the goal
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                distance_to_goal = abs(o['ball'][0] - 1)
                if distance_to_goal < self.shoot_distance_threshold:
                    components['shooting_reward'][i] = 0.5  # Increment if shooting range
                    reward[i] += components['shooting_reward'][i]

            # Effective dribbling
            if o['sticky_actions'][9] == 1:  # Active dribbling
                distance_to_opponents = np.min(np.linalg.norm(o['right_team'] - o['left_team'][o['active']], axis=1))
                if distance_to_opponents < self.dribble_effectiveness_threshold:
                    components['dribbling_reward'][i] = 0.3  # Successful dribbling close to an opponent
                    reward[i] += components['dribbling_reward'][i]

            # Passing effectively
            if any(o['sticky_actions'][6:9]):  # If any of the pass actions are active
                for teammate_pos in o['left_team']:
                    if np.linalg.norm(teammate_pos - o['ball'][:2]) < self.pass_accuracy_threshold:
                        components['passing_reward'][i] += 0.2
                        reward[i] += components['passing_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        info["final_reward"] = sum(modified_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.env.unwrapped.observation()  # Update sticky actions count
        return observation, modified_reward, done, info
