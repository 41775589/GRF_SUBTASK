import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on enhancing defensive actions and quick transitions for counter-attacks.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Configuration for decreased/increased reward based on defensive actions
        self.defensive_reward_increase = 0.1
        self.recovery_position_threshold = -0.5  # Position threshold for effective recovery

    def reset(self):
        """Resets the environment and important variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state to allow resuming the game play."""
        to_pickle['defender_rewards'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Set the internal state to resume the game play."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['defender_rewards']
        return from_pickle

    def reward(self, reward):
        """Customize the rewards to emphasize defensive skills and quick transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["defensive_reward"][rew_index] = 0

            # Strengthening reward for effective defensive recoveries.
            if o['game_mode'] in [2, 3, 4, 5]:  # These modes involve potential ball recoveries
                if o['ball_owned_team'] == 0:
                    # Award a bonus if the player is in a recovering position on the field
                    if o['active'] in o['left_team'] and o['left_team'][o['active']][0] < self.recovery_position_threshold:
                        components["defensive_reward"][rew_index] = self.defensive_reward_increase
                        # Increase the reward with proportion to positioning on the field
                        reward[rew_index] += 1.5 * components["defensive_reward"][rew_index]
            
            # Consider reward decay if the agent is holding ball too long without progression
            if o['ball_owned_team'] == 0 and o['ball'][0] < 0:
                # Decrease reward slightly to encourage quick plays or passes
                reward[rew_index] -= 0.05

        return reward, components

    def step(self, action):
        """Process the environment's response to each action taken by the agents."""
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
