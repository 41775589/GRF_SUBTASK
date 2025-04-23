import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for executing long passes accurately across different field zones."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 5
        self._long_pass_reward = 0.2
        self._active_player_previous = -1
        self._previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._active_player_previous = -1
        self._previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'active_player_previous': self._active_player_previous,
                                                'previous_ball_position': self._previous_ball_position}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._active_player_previous = from_pickle['CheckpointRewardWrapper']['active_player_previous']
        self._previous_ball_position = from_pickle['CheckpointRewardWrapper']['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball = np.array(o['ball'][:2])  # consider only x, y coordinates of the ball
            if self._previous_ball_position is not None:
                ball_distance = np.linalg.norm(ball - self._previous_ball_position)

                # Check for a long pass by evaluating if the controlled player is different 
                # and distance of the ball travelled is significant
                if o['active'] != self._active_player_previous and ball_distance > 0.5:
                    components["long_pass_reward"][rew_index] = self._long_pass_reward
                    reward[rew_index] += components["long_pass_reward"][rew_index]

            # Update for next reward calculation
            self._previous_ball_position = ball
            self._active_player_previous = o['active']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
