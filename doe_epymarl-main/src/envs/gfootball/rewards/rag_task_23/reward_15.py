import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive coordination reward, particularly near the penalty area. 
    The reward is based on the defense team's ability to coordinate and prevent the opponent from 
    advancing into the penalty area."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Regular scoring reward remains untouched
        components = {"base_score_reward": reward.copy()}
        
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward.copy(), components
        
        defense_reward = [0.0, 0.0]
        for i, obs in enumerate(observation):
            
            if obs["game_mode"] == 1 and obs["ball_owned_team"] == 1:
                # Defensive scenario near the penalty area and the ball is owned by the opponents
                own_goal_area = [-1, -0.044, 0.044]  # x, y_min, y_max coordinates of the goal area for the left team
                ball_position = obs['ball'][0:2]  # Ball's x and y position
                
                # Check if the ball is within the y-range of the goal and close to the goal line
                if own_goal_area[1] <= ball_position[1] <= own_goal_area[2] and ball_position[0] < -0.75:
                    # Assuming agents index 0 and 1 are defenders
                    for agent_index in [0, 1]:
                        if np.linalg.norm(observation[agent_index]['left_team'][obs['active']][0:2] - ball_position) < 0.1:
                            # If one of the defenders is within a close range of the ball
                            defense_reward[agent_index] += 0.1
        
        for i in range(len(reward)):
            reward[i] += defense_reward[i]
            components.setdefault(f"defense_reward_agent_{i}", []).append(defense_reward[i])

        return reward, components

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
