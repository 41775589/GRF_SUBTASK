import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that incentivizes accurate shots under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_position_rewards = np.linspace(0.1, 0.5, num=5)
        self.pressure_multiplier = 1.5

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
        components = {
            "base_score_reward": reward.copy(),
            "shooting_position_reward": [0.0],
            "pressure_adjusted_reward": [0.0]
        }

        if observation is None:
            return reward, components

        for o in observation:
            dist_to_goal = abs(o['ball'][0] - 1)  # Distance to opponent's goal along x-axis
            shooting_index = max(0, min(int(dist_to_goal * 5), 4))
            
            # Add reward based on proximity to the goal when shooting
            if o['sticky_actions'][9]:  # Assuming index 9 is the 'Shot' action
                components['shooting_position_reward'][0] = self.shooting_position_rewards[shooting_index]
                reward[0] += components['shooting_position_reward'][0]
               
                # Adding pressure multiplier based on proximity of opponents
                opponents_distance = np.min(np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1))
                pressure = np.exp(-opponents_distance)
                components['pressure_adjusted_reward'][0] = components['shooting_position_reward'][0] * self.pressure_multiplier * pressure
                reward[0] += components['pressure_adjusted_reward'][0]
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        reward, components = self.reward(reward)
        
        # Adding reward details to info for monitoring and debugging purposes
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Reset sticky_actions_counter and update with new counts
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)

        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action if action > self.sticky_actions_counter[i] else self.sticky_actions_counter[i]

        info.update({f"sticky_actions_{i}": self.sticky_actions_counter[i] for i in range(10)})  # Populate action statistics
        return observation, reward, done, info
