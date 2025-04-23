import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages goalkeeper coordination and efficient ball clearance to outfield players."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to adjust the rewards for actions of interest
        self.ball_clearance_reward = 0.1
        self.goalkeeper_backup_reward = 0.15

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
                      "ball_clearance_reward": [0.0] * len(reward),
                      "goalkeeper_backup_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for clearing the ball efficiently to outfield players
            if o['game_mode'] in [2, 3, 4]:  # Game modes representing set pieces like goals, free kicks, corners
                # Assuming that ball_owned_team 0 is the left team including the goalkeeper
                if o['ball_owned_team'] == 0 and \
                   o['ball_owned_player'] >= 0 and \
                   o['right_team_roles'][o['ball_owned_player']] == 0:  # Role 0 for goalkeeper
                    components["ball_clearance_reward"][rew_index] = self.ball_clearance_reward
                    reward[rew_index] += components["ball_clearance_reward"][rew_index]

            # Reward for proper goalkeeper backup behavior
            closest_defender_distance = np.min(np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1))
            if o['right_team_roles'][o['ball_owned_player']] == 0 and closest_defender_distance < 0.1:
                components["goalkeeper_backup_reward"][rew_index] = self.goalkeeper_backup_reward
                reward[rew_index] += components["goalkeeper_backup_reward"][rew_index]

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
