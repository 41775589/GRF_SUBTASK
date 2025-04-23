import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high-quality passes from midfield to create scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment and include the sticky actions counter."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from the pickled data."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Adjust rewards based on the quality of high passes from midfield."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        assert len(reward) == 2  # Expecting rewards for two agents
        base_score_reward = reward.copy()
        high_pass_reward = [0.0, 0.0]
        for i in range(len(reward)):
            o = observation[i]
            if o is None:
                continue
            ball_position = o['ball']
            ball_owned_team = o['ball_owned_team']
            ball_owned_player = o['ball_owned_player']
            active = o['active']

            # Reward condition: check if the active player is a midfielder and owns the ball
            if ball_owned_team == 0 and ball_owned_player == active and o['left_team_roles'][active] in [4, 5, 6]:
                # Check if the ball is in the midfield zone
                if -0.3 <= ball_position[0] <= 0.3:
                    pass_quality = np.linalg.norm(ball_position - o['right_team'][np.argmin(np.linalg.norm(o['right_team'] - ball_position, axis=1))])
                    # Reward high quality passes closer to the opponent's goal with direct trajectory upward
                    if pass_quality < 0.2 and o['ball_direction'][1] > 0.3:
                        high_pass_reward[i] += 0.5

        # Combine components and final reward calculation
        components = {
            'base_score_reward': base_score_reward,
            'high_pass_reward': high_pass_reward
        }
        final_rewards = [base_score_reward[i] + high_pass_reward[i] for i in range(2)]
        
        return final_rewards, components

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
