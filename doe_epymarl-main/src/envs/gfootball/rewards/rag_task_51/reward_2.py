import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper that modifies the rewards for specialized goalkeeper training.
    This includes rewards for shot-stopping, quick reflexes, and promoting quick,
    accurate passes to initiate counter-attacks.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.block_coefficient = 1.0
        self.pass_coefficient = 0.5
        self.reflex_coefficient = 0.3

    def reset(self):
        """
        Reset the environment and the counter for sticky actions.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state for this wrapper along with its environment's state.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state from the given saved state.
        """
        from_pickle = self.env.set_state(state)
        # Restore any internal state if saved previously
        return from_pickle

    def reward(self, reward):
        """
        Modifies the base reward by incorporating goalkeeper-specific behaviors.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "block_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "reflex_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, player_obs in enumerate(observation):
            active_role = player_obs['left_team_roles'][player_obs['active']]
            if active_role == 0:  # Only provide specific rewards if player is a goalkeeper
                ball_dist = np.linalg.norm(player_obs['ball'] - player_obs['left_team'][player_obs['active']])
                opponent_dist = np.min(np.linalg.norm(player_obs['left_team'][player_obs['active']] - player_obs['right_team'], axis=1))
                
                # Reward for blocking (closer ball distances during opponent team attacks)
                if opponent_dist < 0.3 and ball_dist < 0.15:
                    components['block_reward'][rew_index] = self.block_coefficient

                # Reflex reward (successful actions under high pressure)
                if player_obs['sticky_actions'][3] == 1:  # top_right action often a dive
                    components['reflex_reward'][rew_index] = self.reflex_coefficient
                
                # Reward for starting counter-attacks (quick passes after blocking)
                if player_obs['sticky_actions'][9] == 1:  # dribble action for a goalkeeper assumed to be a quick pass
                    components['pass_reward'][rew_index] = self.pass_coefficient

                # Combine all components
                reward[rew_index] = (reward[rew_index] + 
                                     components['block_reward'][rew_index] +
                                     components['pass_reward'][rew_index] +
                                     components['reflex_reward'][rew_index])

        return reward, components

    def step(self, action):
        """
        Take a step in the environment applying the custom rewards and updates sticky actions counter.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_status in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_status
        return observation, reward, done, info
