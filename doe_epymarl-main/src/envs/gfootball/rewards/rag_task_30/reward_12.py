import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes mastering strategic positioning and transitions in gameplay."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reset()

    def reset(self):
        """Reset the wrapper's state at the start of each episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment along with wrapper specific additions."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment and any wrapper specific additions."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Compute and return the modified reward with additional components."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            ball_pos = player_obs['ball']
            team_pos = player_obs['left_team' if rew_index == 0 else 'right_team']
            opposing_team_pos = player_obs['right_team' if rew_index == 0 else 'left_team']

            # Reward strategic positioning: staying between ball and home goal
            home_goal_y = team_pos.mean(axis=0)[1]  # average y-coordinate of own team
            if rew_index == 0:
                if all(player_pos[1] < ball_pos[1] for player_pos in team_pos):  # all players behind the ball
                    components["positioning_reward"][rew_index] = 0.05  # slight reward for good positioning
            else:  # Right team, mirror the condition
                if all(player_pos[1] > ball_pos[1] for player_pos in team_pos):
                    components["positioning_reward"][rew_index] = 0.05 

            # Reward for closing down space making lateral or backward movements to block opponents
            if player_obs['ball_owned_team'] == 1 - rew_index:  # if opposing team owns the ball
                distance_to_opponents = [np.linalg.norm(ball_pos - opp_pos) for opp_pos in opposing_team_pos]
                if min(distance_to_opponents) < 0.1:
                    components["positioning_reward"][rew_index] += 0.1  # reward for effective defensive gap closure

            reward[rew_index] += sum(components[x][rew_index] for x in components)
        
        return reward, components

    def step(self, action):
        """
        Executes a step in the environment, capturing secondary data and adjusting reward.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
