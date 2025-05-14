import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A gym wrapper that focuses on training for long passes and effective sprint usage."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Number of regions the pitch is divided into to create checkpoints
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.1  # Reward granted for each checkpoint reached
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for sticky actions

    def reset(self):
        """Reset sticky action counters and environment state."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the checkpoints state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Reward based on a successfully made long pass and the usage of sprint efficiently."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "checkpoint_reward": [0.0] * len(reward),
            "long_pass_sprint_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        agent_obs = observation[0]  # Assuming single agent scenario

        # Check if the agent used sprint effectively
        if agent_obs['sticky_actions'][8] == 1:  # Assuming 8 is the index for sprint action
            self.sticky_actions_counter[8] += 1
        
        # Calculate the distance the ball was moved towards the opponent's goal during a long pass
        if agent_obs['ball_owned_team'] == 0:  # Assuming 0 is the index for the left team (agent's team)
            if abs(agent_obs['ball_direction'][0]) > 0.1:  # Significant forward motion in x-direction
                ball_start_pos = agent_obs['ball']
                ball_end_pos = ball_start_pos + agent_obs['ball_direction']
                ball_moved_towards_goal = ball_end_pos[0] - ball_start_pos[0]  # Positive if moved towards right goal
                
                # Check if the ball is moved towards opponent's goal
                if ball_moved_towards_goal > 0:
                    components['long_pass_sprint_reward'][0] = self._checkpoint_reward * ball_moved_towards_goal
                    reward[0] += components['long_pass_sprint_reward'][0]

                # Sprint effectiveness in moving towards ball
                if self.sticky_actions_counter[8] >= 1:  # Using sprint
                    reward[0] += self._checkpoint_reward * self.sticky_actions_counter[8]  # Bonus for sprint usage
        
        return reward, components

    def step(self, action):
        """Execute a step using the chosen action."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Reset sticky actions counter after applying them
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
