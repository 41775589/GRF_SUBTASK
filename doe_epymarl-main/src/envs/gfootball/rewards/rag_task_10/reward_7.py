import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward with defensive skill metrics."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'defensive_skill_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Encourage dispossessing the ball or preventing scoring.
            if o['game_mode'] in [0, 4, 5]:  # Normal, Corner, Throw in
                # Ball ownership status rewards/penalties.
                if o['ball_owned_team'] == 0:  # Agent's team owns the ball
                    if o['sticky_actions'][7]:  # Slide action
                        components['defensive_skill_reward'][rew_index] += 0.2
                    if o['sticky_actions'][6] or o['sticky_actions'][4]:  # Stop the ball
                        components['defensive_skill_reward'][rew_index] += 0.1
                elif o['ball_owned_team'] == 1:  # Opponent's team owns the ball
                    components['defensive_skill_reward'][rew_index] -= 0.1

            reward[rew_index] += components['defensive_skill_reward'][rew_index]

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
