import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for specific dribbling skills against a goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_threshold = 0.1
        self.goalkeeper_presence_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_bonus": [0.0] * len(reward),
            "goalkeeper_presence_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]
            goalkeeper_position = o['right_team'][0]  # Assuming index 0 is the goalkeeper

            # Check if the active player has the ball, encourage dribbling.
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Dribbling bonus encouraged near the goalkeeper
                distance_to_goalkeeper = np.linalg.norm(ball_pos - goalkeeper_position)
                if distance_to_goalkeeper < self.dribbling_threshold:
                    components["dribbling_bonus"][rew_index] = 0.5
                    reward[rew_index] += components["dribbling_bonus"][rew_index]

                # Bonus for keeping the ball under pressure near the goalkeeper
                if o['game_mode'] == 0 and distance_to_goalkeeper < 0.2:
                    components["goalkeeper_presence_bonus"][rew_index] = self.goalkeeper_presence_reward
                    reward[rew_index] += components["goalkeeper_presence_bonus"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle
