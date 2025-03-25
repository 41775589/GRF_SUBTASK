import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for good defensive positions and transitions to counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize counter for recording action frequencies
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward parameters
        self.counter_attack_bonus = 1.0  # Bonus for moving towards a counter-attack setup
        self.defensive_stance_bonus = 0.5  # Bonus for taking good defensive positions
        self.checkpoints = [0.0, 0.25, 0.5, 0.75, 1.0]  # Positions for counter-attack bonuses

    def reset(self):
        """Resets the environment and clears action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Returns the state with wrapper-specific values."""
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        """Sets the state with wrapper-specific values."""
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return self.env.set_state(state)

    def reward(self, reward):
        """Enhances the reward based on defensive actions and transitions to counter-attacks."""
        observation = self.env.unwrapped.observation()

        # Initialize component dictionary
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        # If no observation available, return the original reward
        if observation is None:
            return reward, components

        # Process observations for both agents
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o:
                # Reward for defensive stances
                if o['ball_owned_team'] == 0: # Classifier Left team is on defense
                    components["defensive_reward"][rew_index] = self.defensive_stance_bonus

                # Reward for transitioning to counter-attack
                ball_position = o['ball'][0]  # Get ball's X position
                for checkpoint in self.checkpoints:
                    if ball_position > checkpoint:
                        components["transition_reward"][rew_index] += self.counter_attack_bonus

                # Calculate total reward for this agent
                total_reward = reward[rew_index] + components["defensive_reward"][rew_index] + components["transition_reward"][rew_index]
                reward[rew_index] = total_reward

        return reward, components

    def step(self, action):
        """Steps through the environment, applies the reward wrapper, and records information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Aggregate final reward and component values in the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions applied (last step)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
