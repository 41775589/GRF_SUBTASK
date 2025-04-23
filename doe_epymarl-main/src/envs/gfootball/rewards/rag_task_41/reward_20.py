import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on attacking skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Zones on the field that promote attacking training elements
        self.zones = {
            'close': 0.3,  # Close to opponent's goal
            'midfield': 0.6,  # Midfield area
            'distant': 0.9   # Distant area from opponent's goal
        }
        self.zone_rewards = {
            'close': 1,
            'midfield': 0.5,
            'distant': 0.2
        }

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
                      "zone_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            agent_obs = observation[i]
            ball_x = agent_obs['ball'][0]
            team_possession = agent_obs['ball_owned_team'] == 1

            if team_possession and ball_x > 0:  # Only reward when in opponent's half
                zone = None
                for key, distance in self.zones.items():
                    if ball_x > distance:
                        zone = key
                        break
                if zone:
                    zone_rew = self.zone_rewards[zone]
                    reward[i] += zone_rew
                    components["zone_reward"][i] += zone_rew

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
