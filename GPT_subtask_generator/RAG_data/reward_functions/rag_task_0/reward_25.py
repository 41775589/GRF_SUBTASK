import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense, multi-faceted reward designed for offensive training, focusing on dribbling, shooting, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.dribbling_control = np.zeros(10, dtype=float)
        self.passing_precision = np.zeros(10, dtype=float)
        self.shooting_accuracy = 0.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.dribbling_control.fill(0)
        self.passing_precision.fill(0)
        self.shooting_accuracy = 0.0
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribbling_control'] = self.dribbling_control
        to_pickle['passing_precision'] = self.passing_precision
        to_pickle['shooting_accuracy'] = self.shooting_accuracy
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribbling_control = from_pickle.get('dribbling_control', np.zeros(10, dtype=float))
        self.passing_precision = from_pickle.get('passing_precision', np.zeros(10, dtype=float))
        self.shooting_accuracy = from_pickle.get('shooting_accuracy', 0.0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward,
            "dribbling_control": self.dribbling_control.copy(),
            "passing_precision": self.passing_precision.copy(),
            "shooting_accuracy": self.shooting_accuracy
        }
        
        # Perform dribbling reward calculations
        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['sticky_actions'][9]:  # Assuming dribble action index is 9
                self.dribbling_control[i] += 0.05
            reward[i] += self.dribbling_control[i]

        # Perform passing reward calculations
        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['game_mode'] in [2, 5]:  # Assuming kick off and throw in favor passes
                self.passing_precision[i] += 0.1
            reward[i] += self.passing_precision[i]

        # Perform shooting reward calculations
        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['game_mode'] == 6:  # Assuming penalty kick is a shot event
                self.shooting_accuracy += 0.2
            reward[i] += self.shooting_accuracy
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
