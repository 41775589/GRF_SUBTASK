import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering the technique of short passing under defensive pressure.
    The reward focuses on ball retention and effective distribution to aid in defensive stability and counter-attack transitions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy = 0.1  # Reward for accurate short passes
        self.defensive_pressure = 0.05  # Additional reward under defensive pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Extract the necessary state if needed, here just an example
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_accuracy": len(reward) * [0.0],
                      "defensive_pressure": len(reward) * [0.0]}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for successful short passes
            if 'short_pass' in o['sticky_actions']:
                components["pass_accuracy"][rew_index] = self.pass_accuracy * sum(o['sticky_actions'][0:2])

                # Additional reward if ball possession is maintained under pressure
                if o['ball_owned_team'] == 1 and np.any(o['left_team_tired_factor'] > 0.5):
                    components["defensive_pressure"][rew_index] = self.defensive_pressure
                    components["pass_accuracy"][rew_index] += self.defensive_pressure

            # Update the reward for the current index
            reward[rew_index] += components["pass_accuracy"][rew_index] + components["defensive_pressure"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        # Reset the sticky counter if needed or update it based on current observation
        if obs:
            self.sticky_actions_counter.fill(0)
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action
            
        return observation, reward, done, info
