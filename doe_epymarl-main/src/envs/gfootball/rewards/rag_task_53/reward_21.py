import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that introduces rewards for maintaining ball control under pressure, distributing the ball efficiently,
    and making strategic plays across the field in various zones.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.control_zones = np.linspace(-1, 1, 5)  # Dividing the field into 5 zones
        self.zone_rewards = np.linspace(0.1, 0.5, 5)  # Incremental rewards for ball control in each zone
        self.pressure_reward = 0.2  # Reward for maintaining control under pressure
        self.opponents_close_threshold = 0.1  # Distance threshold to consider opponents close (applying pressure)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_rewards = reward.copy()  # Base rewards from the environment
        components = {
            "base_score_reward": reward.copy(),
            "zone_rewards": [0.0] * len(reward),
            "pressure_rewards": [0.0] * len(reward)
        }

        if observation is None:
            return new_rewards, components

        for i, obs in enumerate(observation):
            ball_x = obs['ball'][0]
            ball_control = (obs['ball_owned_team'] == 0) and (obs['ball_owned_player'] == obs['active'])
            
            if ball_control:
                # Calculate which zone the ball is in and assign the appropriate reward
                zone_index = np.digitize(ball_x, self.control_zones, right=True)
                components["zone_rewards"][i] = self.zone_rewards[zone_index]
                new_rewards[i] += components["zone_rewards"][i]

                # Check proximity of opponents to apply pressure rewards
                opponents_positions = obs['right_team']
                player_position = obs['left_team'][obs['active']]
                distances = np.sqrt(np.sum(np.square(opponents_positions - player_position), axis=1))
                pressure = np.any(distances < self.opponents_close_threshold)

                if pressure:
                    components["pressure_rewards"][i] = self.pressure_reward
                    new_rewards[i] += components["pressure_rewards"][i]

        return new_rewards, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter.tolist(),
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_data = from_pickle.get('CheckpointRewardWrapper', {})
        self.sticky_actions_counter = np.array(saved_data.get("sticky_actions_counter", [0]*10))
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
