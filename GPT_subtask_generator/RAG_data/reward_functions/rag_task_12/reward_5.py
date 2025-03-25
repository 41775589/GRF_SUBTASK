import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward based on the agent's capability to handle midfield and defensive transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_counter = 0
        self.long_pass_counter = 0
        self.dribble_counter = 0
        self.sprint_counter = 0
        self.stop_sprint_counter = 0
        self.last_ball_position = None

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.high_pass_counter = 0
        self.long_pass_counter = 0
        self.dribble_counter = 0
        self.sprint_counter = 0
        self.stop_sprint_counter = 0
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "high_pass_counter": self.high_pass_counter,
            "long_pass_counter": self.long_pass_counter,
            "dribble_counter": self.dribble_counter,
            "sprint_counter": self.sprint_counter,
            "stop_sprint_counter": self.stop_sprint_counter,
            "last_ball_position": self.last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        data = from_pickle['CheckpointRewardWrapper']
        self.high_pass_counter = data["high_pass_counter"]
        self.long_pass_counter = data["long_pass_counter"]
        self.dribble_counter = data["dribble_counter"]
        self.sprint_counter = data["sprint_counter"]
        self.stop_sprint_counter = data["stop_sprint_counter"]
        self.last_ball_position = data["last_ball_position"]
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Evaluate performance based on important actions
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']

            if self.last_ball_position is not None and (o['ball'][0] - self.last_ball_position[0]) != 0:
                if sticky_actions[7] and o['ball_owned_team'] == 0:  # High Pass
                    self.high_pass_counter += 5
                if sticky_actions[9] and o['ball_owned_team'] == 0:  # Long Pass
                    self.long_pass_counter += 7
                if sticky_actions[1]:  # Sprint
                    self.sprint_counter += 2
                if not sticky_actions[1]:  # Stop Sprint
                    self.stop_sprint_counter += 1
                if sticky_actions[8]:  # Dribble
                    self.dribble_counter += 3

            reward[rew_index] += (
                self.high_pass_counter +
                self.long_pass_counter +
                self.dribble_counter +
                self.sprint_counter +
                self.stop_sprint_counter
            ) / 100  # Normalize reward

            components["high_pass_reward"] = self.high_pass_counter
            components["long_pass_reward"] = self.long_pass_counter
            components["dribble_reward"] = self.dribble_counter
            components["sprint_reward"] = self.sprint_counter
            components["stop_sprint_reward"] = self.stop_sprint_counter

            self.last_ball_position = o['ball'][:2]  # Update last ball position
        
        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return obs, reward, done, info
