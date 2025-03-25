import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic reward based on counter-attacking efficiency after ball recovery."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Monitoring each agent's sticky actions
        # Parameters for counter-attacking rewards
        self.ball_recovery_position = {}
        self.counter_attack_efficiency_gain = 2.5

    def reset(self):
        """Reset the state of the environment and wrapper state variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_recovery_position = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state and include the wrapper-specific state."""
        to_pickle['ball_recovery_position'] = self.ball_recovery_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the current state, recovering wrapper-specific state."""
        from_pickle = self.env.set_state(state)
        self.ball_recovery_position = from_pickle.get('ball_recovery_position', {})
        return from_pickle

    def reward(self, reward):
        """Modify the environment's reward based on how effectively agents initiate counter-attacks."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components
        
        components["counter_attack_reward"] = [0.0] * len(reward)

        for idx, o in enumerate(observation):
            if o['game_mode'] == 5 and self.env.previous_game_mode != 5:  # Recovering the ball scenario
                self.ball_recovery_position[idx] = o['ball']

            if o['game_mode'] == 2 and o['ball_owned_team'] == 0:  # Counter attack scenario
                initial_pos = self.ball_recovery_position.get(idx)
                if initial_pos:
                    current_pos = o['ball']
                    distance_advanced = current_pos[0] - initial_pos[0]  # Improve this line for correctness per environment perspective
                    if distance_advanced > 0:
                        components["counter_attack_reward"][idx] = self.counter_attack_efficiency_gain * distance_advanced
                        reward[idx] += components["counter_attack_reward"][idx]

        return reward, components
        
    def step(self, action):
        """Step through environment, adjust rewards, and provide additional debugging information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += int(action)
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
