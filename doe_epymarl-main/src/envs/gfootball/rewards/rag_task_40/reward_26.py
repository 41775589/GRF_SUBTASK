import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to enhance a defensive unit's capability to handle direct attacks by
    incentivizing confrontational defense and strategic positioning for counterattacks.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_bonus_coefficient = 0.1
        self.confrontation_bonus_coefficient = 0.2

    def reset(self):
        """
        Reset the environment state and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the wrapped environment state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the wrapped environment state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on the defensive actions of the players.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward),
                      "confrontational_defense_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for being in good defensive positioning
            if o['ball_owned_team'] != 1:  # Only reward if the opposing team owns the ball
                # Calculate distance from the ball
                distance_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']][:2])
                if distance_to_ball < 0.3:  # Close proximity to the ball leads to higher reward
                    components["positional_reward"][rew_index] = self.position_bonus_coefficient / distance_to_ball
            
            # Reward for confrontational actions (high-pressure defense)
            def is_confrontational(action, action_set):
                confrontational_actions = ['slide', 'high_pressure']
                return any(action in action_set.get(name, "") for name in confrontational_actions)

            current_action_set = self.env.unwrapped.get_action_set()
            if is_confrontational(o['active'], current_action_set):
                components["confrontational_defense_reward"][rew_index] = self.confrontation_bonus_coefficient

            # Update total reward for this player
            reward[rew_index] += components["positional_reward"][rew_index] + \
                                 components["confrontational_defense_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take a step using the specified action, calculate reward, and return observation, reward,
        done, and info.
        """
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
