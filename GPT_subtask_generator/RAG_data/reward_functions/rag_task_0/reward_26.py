import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive strategies: shooting, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_achievements = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_achievements = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.offensive_achievements
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.offensive_achievements = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            if 'ball_owned_team' not in o:
                continue
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                # Check shooting, dribbling, and long passes
                if 'action' in o and o['action'] in {'shot', 'dribble', 'long_pass'}:
                    achievement_key = (idx, o['action'])
                    if achievement_key not in self.offensive_achievements:
                        # Encourage trying different effective offensive skills
                        components['offensive_bonus'][idx] = 0.1 + reward[idx]
                        reward[idx] += components['offensive_bonus'][idx]
                        self.offensive_achievements[achievement_key] = True

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
