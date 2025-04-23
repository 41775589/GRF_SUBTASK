import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for successful tackles and possession regain."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions
    
    def reset(self):
        """Reset the sticky actions counter on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include the wrapper's state in the environment's state serialization."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment and update the wrapper's state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Reward function that emphasizes successful tackles and possession improvements."""
        observation = self.env.unwrapped.observation()

        # Prepare reward components.
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Reward mechanics for successful tackles and possession regain.
        for rew_index, obs in enumerate(observation):
            if 'ball_owned_team' in obs:
                components['tackle_reward'][rew_index] = 0.0
                if obs['ball_owned_team'] == 0: # Assuming the controlled team is '0'
                  
                    # Reward a successful tackle
                    if obs['game_mode'] in (3, 5, 6):  # FreeKick, Corner, Penalty
                        components['tackle_reward'][rew_index] = 0.1

                    # Reward regaining possession not following a penalty/foul
                    if 'ball_owned_player' in obs:
                        components['tackle_reward'][rew_index] += 0.2 if obs['ball_owned_player'] == obs['active'] else 0

            # Calculate final reward for each agent.
            reward[rew_index] += components['tackle_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Apply actions, step the environment, and apply the reward transformation."""
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
