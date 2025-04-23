import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances team synergy during possession changes, 
    with an emphasis on tactical positioning and timing between offensive and defensive phases."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track how often sticky actions are taken
        self.previous_ball_owner = None  # Track the last team that owned the ball

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the persisted state along with internal variables."""
        state = self.env.get_state(to_pickle)
        state['previous_ball_owner'] = self.previous_ball_owner
        return state

    def set_state(self, from_pickle):
        """Sets the state based on unpickled data."""
        state = self.env.set_state(from_pickle)
        self.previous_ball_owner = state.get('previous_ball_owner', None)
        return state

    def reward(self, reward):
        """Custom reward function focusing on promoting transitions and precise positioning."""
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),  # Existing reward without wrapper modifications
            "possession_change_reward": [0.0] * len(reward),
            "positional_advantage_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            current_ball_owner = o['ball_owned_team']

            # Reward for ball possession changes to simulate defensive to offensive transitions
            if self.previous_ball_owner is not None and self.previous_ball_owner != current_ball_owner and current_ball_owner != -1:
                components["possession_change_reward"][rew_index] = 0.3
                reward[rew_index] += components["possession_change_reward"][rew_index]

            # Reward based on player's strategic positioning during offensive phase
            if current_ball_owner == 0:  # Say 0 is our team
                player_pos = o['left_team'][o['active']]
                goal_pos = [1, 0]  # Assuming the goal is at (1,0)
                # Calculating Euclidean distance to the goal
                distance_to_goal = np.sqrt((player_pos[0] - goal_pos[0]) ** 2 + (player_pos[1] - goal_pos[1]) ** 2)
                components["positional_advantage_reward"][rew_index] = (1 - distance_to_goal) * 0.1
                reward[rew_index] += components["positional_advantage_reward"][rew_index]

            self.previous_ball_owner = current_ball_owner

        return reward, components

    def step(self, action):
        """Step environment, modify reward and append additional information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)

        # Tracking sticky actions for each agent in obversation
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action

        return observation, reward, done, info
