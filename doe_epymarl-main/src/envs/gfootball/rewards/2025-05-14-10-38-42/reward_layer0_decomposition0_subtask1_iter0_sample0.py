import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward focused on defensive playing,
    ball positioning, and passing effectiveness for enhanced teamwork and disruption."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_reward = 0.1
        self.positioning_bonus = 0.05

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_count': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', {}).get('sticky_actions_count', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        """Customize reward here by considering the observations."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        for reward_index, observation in enumerate(observation):
            # Encourage agent to keep possession as a defender
            if observation.get('ball_owned_team') == observation.get('active'):
                # Check if the ball is passed successfully by looking at the change
                if 'ball_owned_player' in observation and observation['ball_owned_player'] == observation['active']:
                    reward[reward_index] += self.pass_accuracy_reward

            # Reward for good positioning related to the ball and opposition players
            my_team_position = observation['left_team'] if observation['active'] in observation['left_team'] else observation['right_team']
            opponent_team_position = observation['right_team'] if observation['active'] in observation['left_team'] else observation['left_team']
            ball_position = observation['ball']

            # Encourage staying between ball and goal (goal-line defense) - defensive positioning
            goal_position = [1, 0] if observation['active'] in observation['right_team'] else [-1, 0]
            if self.is_defensive_positioning(my_team_position, ball_position, goal_position):
                reward[reward_index] += self.positioning_bonus
        
        return reward, components

    def is_defensive_positioning(self, my_position, ball_position, goal_position):
        """Check if agent is between the ball and the goal post."""
        return np.linalg.norm(np.array(ball_position) - np.array(my_position)) + \
               np.linalg.norm(np.array(goal_position) - np.array(my_position)) < np.linalg.norm(np.array(goal_position) - np.array(ball_position))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
