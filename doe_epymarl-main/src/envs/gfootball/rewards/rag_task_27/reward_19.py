import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward specifically tailored for defensive skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky actions counter
    
    def reset(self):
        """Reset sticky actions and other necessary states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the state of the checkpoints."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state from a pickle."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Compute reward taking into account defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_position_reward": [0.0] * len(reward),
            "interception_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = components["base_score_reward"][rew_index]

            # Encouraging defensive positioning: being in the right place
            if o['ball_owned_team'] == 1:  # Opponent has the ball
                # Calculate distance from the ball to each player
                for idx, player_pos in enumerate(o['left_team']):
                    ball_pos = o['ball']
                    distance = np.sqrt((player_pos[0] - ball_pos[0])**2 + (player_pos[1] - ball_pos[1])**2)
                    # Reward for being close to the ball when the other team has possession
                    if distance < 0.1:  # Threshold distance to be considered as good defensive position
                        components["defensive_position_reward"][rew_index] += 0.05

            # Reward for interceptions
            if 'ball_owned_player' in o:
                # Player gains control over the ball from the opponent
                if o['previous_ball_owned_team'] == 1 and o['ball_owned_team'] == 0:
                    components["interception_reward"][rew_index] += 0.2

            # Combine rewards
            reward[rew_index] = base_reward + components["defensive_position_reward"][rew_index] + components["interception_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Execute a step in the environment, taking care of reward modifications."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        # Include reward components in the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Record sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = max(self.sticky_actions_counter[i], action)
        return observation, reward, done, info
