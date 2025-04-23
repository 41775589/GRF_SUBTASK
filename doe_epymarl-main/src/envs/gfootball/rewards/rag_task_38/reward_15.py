import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom wrapper to reward agents for successfully initiating counterattacks using long passes and quick transitions.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Custom state for unwrapping could be added here
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Custom state for setting could be added here
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counterattack_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if the team has just regained ball possession
            if ('ball_owned_team' not in o or o['ball_owned_team'] != 1):
                continue
            
            # We assume a transition from defense to attack involves a big change in ball position
            # This is a bit simplified, can be further enhanced by more precise conditions
            ball_x_change = np.abs(np.diff(o['ball'][0]))
            
            # Transition reward: assumption that transition occurs when there's a significant movement of the ball
            if ball_x_change > 0.5:  # Assume a significant x-direction change indicates a big counterattack pass
                components["counterattack_reward"][rew_index] = 0.5  # Custom chosen reward

            # Update the base reward logic using counterattack reward
            reward[rew_index] += components["counterattack_reward"][rew_index]

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
