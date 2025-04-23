import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for clearing the ball from defensive zones under pressure.
    This helps in training agents to effectively clear the ball to ensure safety under pressure.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.defensive_zone_threshold = -0.5  # Defensive half of the field
        self.pressure_radius = 0.1  # Radius to check for opposing players
        self.clearance_reward = 0.3  # Reward for clearing the ball under pressure
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clearance_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the ball is in the defensive half and our team owns it
            if o['ball'][0] <= self.defensive_zone_threshold and o['ball_owned_team'] == 0:
                player_pos = o['left_team'][o['active']]
                pressure = False

                # Check for opposing players within a certain radius
                for opponent_pos in o['right_team']:
                    distance = np.linalg.norm(opponent_pos - player_pos)
                    if distance < self.pressure_radius:
                        pressure = True
                        break

                if pressure:
                    # Reward for clearing the ball under pressure
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += components["clearance_reward"][rew_index]

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
