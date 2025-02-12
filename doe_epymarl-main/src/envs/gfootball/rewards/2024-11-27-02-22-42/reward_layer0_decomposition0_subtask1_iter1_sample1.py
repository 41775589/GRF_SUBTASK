import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards for a soccer defense task based on strategic actions and position control."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Control the importance of each reward component
        self.positioning_reward_factor = 0.3
        self.action_reward_factor = 0.7
        self.base_score_coefficient = 1.0

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()  # Initial reward from the environment
        defensive_action_reward = [0.0] * len(reward)
        position_control_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {
                'base_score_reward': base_score_reward,
                'defensive_action_reward': defensive_action_reward,
                'position_control_reward': position_control_reward
            }

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # Check team ownership of the ball
                # Using defensive action indices assuming: {0: 'Slide', 1: 'Short Pass', 3: 'Sprint', 7: 'Stop Sprint'}
                actions = o['sticky_actions']
                active_defensive_actions = actions[0] or actions[1] or actions[3] or actions[7]
                if active_defensive_actions:
                    defensive_action_reward[rew_index] += self.action_reward_factor
                
                # Position control - incentive to position ourselves between ball and goal
                goal_position = np.array([1, 0])  # Assuming goal is at (1, 0) for the enemy team
                player_positions = o['left_team']
                ball_position = np.array(o['ball'][:2])
                for position in player_positions:
                    dist_to_ball = np.linalg.norm(np.array(position) - ball_position)
                    dist_to_goal = np.linalg.norm(np.array(position) - goal_position)
                    position_control_reward[rew_index] += (self.positioning_reward_factor / (dist_to_ball + 0.1)) 
                    position_control_reward[rew_index] += (self.positioning_reward_factor / (dist_to_goal + 0.1))

            # Calculate final reward with all components
            reward[rew_index] = (
                self.base_score_coefficient * base_score_reward[rew_index] +
                defensive_action_reward[rew_index] +
                position_control_reward[rew_index]
            )

        return reward, {
            'base_score_reward': base_score_reward, 
            'defensive_action_reward': defensive_action_reward,
            'position_control_reward': position_control_reward
        }

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
