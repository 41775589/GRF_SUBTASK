import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages mastering close-range attacks with precision in shooting and effective dribbling against goalkeepers."""

    def __init__(self, env):
        super().__init__(env)
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
                      "precision_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for scoring, scaled based on distance to keeper and precision for close-range
            if o['game_mode'] == 6:  # Penalty mode might indicate close range
                pos_ball = o['ball'][:2]
                keeper_pos = o['right_team'][0] if o['ball_owned_team'] == 0 else o['left_team'][0]

                dist_to_keeper = np.linalg.norm(pos_ball - keeper_pos)
                precision_scale = max(0, 1 - dist_to_keeper)
                components['precision_reward'][rew_index] = precision_scale * 0.5
                reward[rew_index] += components['precision_reward'][rew_index]

            # Reward for dribbling effectively
            if 'sticky_actions' in o:
                dribbling = o['sticky_actions'][9]  # action_dribble index
                components['dribble_reward'][rew_index] = dribbling * 0.1
                reward[rew_index] += components['dribble_reward'][rew_index]

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                if act:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
