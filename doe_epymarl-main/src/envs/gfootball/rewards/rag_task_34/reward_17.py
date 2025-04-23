import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward by focusing on shot precision and effectiveness against the goalkeeper."""
    
    def __init__(self, env):
        super().__init__(env)
        self.shot_precision_reward = 0.3  # Reward for precision in shots
        self.dribble_effectiveness_reward = 0.2  # Reward for effective dribbling close to the goal
        self.goalkeeper_proximity_reward = 0.5  # Additional reward when close to the goalkeeper
        self.shot_on_target_reward = 2.0  # Reward when a shot is on target
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions

    def reset(self):
        """Resets the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Passes the state getting functionality to the underlying environment."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Passes the state setting functionality to the underlying environment."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Calculates a modified reward based on shot precision and dribbling effectiveness."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward,
            "shot_precision_reward": [0.0] * len(reward),
            "dribble_effectiveness_reward": [0.0] * len(reward),
            "goalkeeper_proximity_reward": [0.0] * len(reward),
            "shot_on_target_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for index, rew in enumerate(reward):
            o = observation[index]
            if o['ball_owned_team'] == 0:  # We assume team 0 is the agent's team
                # Calculate the proximity to the goalkeeper (goalkeeper is typically index 0 in 'left_team_roles')
                goalie_position = o['left_team'][0]
                player_position = o['right_team'][o['active']]
                distance_to_goalie = np.linalg.norm(goalie_position - player_position)

                # If close to the goalkeeper, award precision reward
                if distance_to_goalie < 0.1:
                    components["goalkeeper_proximity_reward"][index] = self.goalkeeper_proximity_reward

                # Check shot precision by looking at the direction and position of the ball relative to the goal
                if o['ball_direction'][0] > 0:  # Assuming ball moving towards goal at x > 0
                    goal_x = 1  # x-coordinate of opponent's goal
                    distance_to_goal = abs(goal_x - o['ball'][0])
                    if distance_to_goal < 0.1:  # Ball is close to the goal axis
                        components["shot_precision_reward"][index] = self.shot_precision_reward
                        components["shot_on_target_reward"][index] = self.shot_on_target_reward

                # Reward based on effective dribbling actions when near goalkeeper
                if 'action_dribble' in o['sticky_actions'] and distance_to_goalie < 0.2:
                    components['dribble_effectiveness_reward'][index] = self.dribble_effectiveness_reward
            
            # Summing up the rewards with components
            reward[index] += sum(components[e][index] for e in components)

        return reward, components

    def step(self, action):
        """Advances the environment by one step and computes the customized reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
