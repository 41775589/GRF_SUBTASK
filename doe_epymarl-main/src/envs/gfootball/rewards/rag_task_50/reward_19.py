import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards long passes that accurately connect different areas of the playfield.
    It incentivizes strategic vision, timing, and precision in ball distribution.
    """

    def __init__(self, env, pass_zones=5):
        super().__init__(env)
        self.pass_zones = pass_zones
        self.initial_player_positions = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owner_position = None
        self.passing_reward = 0.5

    def reset(self):
        """
        Reset the environment and clear the data used to compute rewards.
        """
        self.initial_player_positions = []
        self.sticky_actions_counter.fill(0)
        self.last_ball_owner_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the internal state including all necessary checkpoints.
        """
        to_pickle['CheckpointRewardWrapper'] = {
            'initial_player_positions': self.initial_player_positions,
            'last_ball_owner_position': self.last_ball_owner_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the internal state from a previous saved state.
        """
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle['CheckpointRewardWrapper']
        self.initial_player_positions = saved_state['initial_player_positions']
        self.last_ball_owner_position = saved_state['last_ball_owner_position']
        return from_pickle

    def reward(self, reward):
        """
        Enhance the reward based on the long passes that connect different playfield areas.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward}
        
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0, 0.0]
        }

        for i, obs in enumerate(observation):
            # Ensure the active player from the team is in possession of the ball
            if obs['ball_owned_team'] == obs['active'] and obs['ball_owned_team'] != -1:
                
                current_ball_position = obs['ball'][:2]  # Ignore z-coordinate
                if self.last_ball_owner_position is not None:
                    distance_moved = np.linalg.norm(current_ball_position - self.last_ball_owner_position)

                    # Reward successful long passes that markedly change the ball's position across the field
                    if distance_moved > 0.5:  # Threshold as placeholder; define according to environment specifics
                        components["long_pass_reward"][i] = self.passing_reward
                        reward[i] += self.passing_reward

                # Update the last ball owner position for further tracking
                self.last_ball_owner_position = current_ball_position

        return reward, components

    def step(self, action):
        """
        Execute a step using the given action, obtaining observations and augmenting the reward.
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
