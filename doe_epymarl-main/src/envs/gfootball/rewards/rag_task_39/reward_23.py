import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for clearing the ball under pressure in defensive zones."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        return to_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_clearance_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # We aim to reward clearances from the defensive third, under pressure
            if o['ball_owned_team'] == 0 and o['left_team'][o['active']][0] < -0.5 and o['game_mode'] == 0:
                # Check if there are opposing players nearby
                opponent_distances = np.linalg.norm(o['right_team'] - o['left_team'][o['active']], axis=1)
                pressure = np.any(opponent_distances < 0.1)
                
                # Reward if the ball has been cleared effectively to a safer zone
                if pressure and o['ball_direction'][0] > 0.1:
                    components["defensive_clearance_reward"][rew_index] = 0.5
                    reward[rew_index] += components["defensive_clearance_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action_status in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action_status

        return observation, reward, done, info
