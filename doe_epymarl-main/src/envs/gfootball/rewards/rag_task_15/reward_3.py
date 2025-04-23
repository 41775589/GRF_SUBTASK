import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for successful long passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.long_pass_threshold = 0.4  # Threshold for considering a pass as long
        self.accuracy_reward = 1.0       # Reward for accurate long passes
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Work with the previous position and new position of the ball
            last_ball_pos = o['ball']
            last_ball_owner_team = o['ball_owned_team']
            ball_travel = np.linalg.norm(o['ball_direction'][:2])

            # Check if a long pass was executed
            if ('ball_owned_team' in o and o['ball_owned_team'] == 1 and  # The right team makes the pass
                'ball_owned_player' in o and o['active'] == o['designated'] and  # Active player, the one controlling the action
                ball_travel > self.long_pass_threshold and  # The ball traveled a long distance
                last_ball_owner_team == 1):  # was owned by the same team before the action
                    components['long_pass_reward'][rew_index] = self.accuracy_reward
                    reward[rew_index] += components['long_pass_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
