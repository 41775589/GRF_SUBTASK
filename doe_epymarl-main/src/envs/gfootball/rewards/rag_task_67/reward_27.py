import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on passing, dribbling, and ball control skills.
    
    This reward encourages the agent to efficiently transition from defense to 
    attack by advancing the ball towards the opponent's goal through controlled plays like short and long passes and dribbles.
    """

    def __init__(self, env):
        """Initializes the CheckpointRewardWrapper."""
        super(CheckpointRewardWrapper, self).__init__(env)
        self.checkpoints_reached = 0
        self.total_distance_moved = 0
        self.previous_ball_position = np.array([0, 0])
        self.possession_change_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the reward wrapper state for a new episode."""
        self.checkpoints_reached = 0
        self.total_distance_moved = 0
        self.previous_ball_position = np.array([0, 0])
        self.possession_change_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Modifies the reward based on ball control and movement skills."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": copy.deepcopy(reward), "transition_reward": np.zeros(len(reward))}

        for i in range(len(reward)):
            o = observation[i]
            # Encourage advancing the ball towards the opponent's half
            if o['ball_owned_team'] == 0:  # Assuming the agent's team is '0'
                current_ball_position = np.array(o['ball'][:2])
                movement_vector = current_ball_position - self.previous_ball_position
                distance_moved = np.linalg.norm(movement_vector)
                self.total_distance_moved += distance_moved

                # Calculate reward based on possession and forward movement
                if self.previous_ball_position[0] > current_ball_position[0]:  # Moving forward
                    forward_reward = distance_moved * 0.1  # Scale for reward balancing
                    components['transition_reward'][i] += forward_reward

                self.previous_ball_position = current_ball_position

            # Bonus for keeping possession under pressure
            if o['game_mode'] in [3, 4]:  # Free kick or corner
                components['transition_reward'][i] += 0.5  # Static bonus for maintaining possession
                
            # Count possession changes as negative rewards
            if o['ball_owned_team'] != -1 and o['ball_owned_team'] != self.possession_change_counter:
                components['transition_reward'][i] -= 0.2  # Penalty for losing the ball
                self.possession_change_counter = o['ball_owned_team']

            reward[i] += components['transition_reward'][i]

        return reward, components

    def step(self, action):
        """Steps through the environment with the given action."""
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
