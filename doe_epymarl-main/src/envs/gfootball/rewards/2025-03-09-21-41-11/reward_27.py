import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on offensive strategies including shooting accuracy,
    dribbling skills to evade opponents, and effective passing.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_bonus = 0.1
        self.shoot_bonus = 0.2
        self.dribble_bonus = 0.05

    def reset(self):
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
                      "pass_bonus": [0.0] * len(reward),
                      "shoot_bonus": [0.0] * len(reward),
                      "dribble_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            if o['game_mode'] in [0, 2]:  # Only apply in Normal and FreeKick game modes
                if o['sticky_actions'][9] == 1:  # action_dribble
                    components['dribble_bonus'][rew_index] = self.dribble_bonus
                    reward[rew_index] += self.dribble_bonus

                if 'ball_owned_team' in o and o['ball_owned_team'] == o['active'] and \
                   'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > 0:
                    if o['sticky_actions'][8] == 1:  # action_sprint
                        components['pass_bonus'][rew_index] = self.pass_bonus
                        reward[rew_index] += self.pass_bonus

                goal_distance = np.abs(o['ball'][0] - 1)  # Assuming playing towards right goal at x=1
                if goal_distance < 0.2 and o['ball_owned_team'] == o['active']:
                    components['shoot_bonus'][rew_index] = self.shoot_bonus
                    reward[rew_index] += self.shoot_bonus

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
