import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shooting practice focusing on accuracy and power."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.last_shot_position = None
        self.shots_on_goal = 0  # Count of shots on goal
        self.scored_goals = 0   # Count of scored goals
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To track sticky actions
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_shot_position = None
        self.shots_on_goal = 0
        self.scored_goals = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {
            'last_shot_position': self.last_shot_position,
            'shots_on_goal': self.shots_on_goal,
            'scored_goals': self.scored_goals
        }
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle.get('CheckpointRewardWrapper', {})
        self.last_shot_position = wrapper_state.get('last_shot_position', None)
        self.shots_on_goal = wrapper_state.get('shots_on_goal', 0)
        self.scored_goals = wrapper_state.get('scored_goals', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_reward = reward.copy()
        shooting_accuracy_reward = [0.0]
        position = observation['right_team'][observation['active'][1]]
        
        if observation['ball_owned_team'] == 1:  # if the right team owns the ball
            if observation['game_mode'] in [3, 4]:  # Shot attempted
                if self.last_shot_position is not None:
                    # Calculate distance from the goal (y = 0 at goal level)
                    distance = np.linalg.norm(np.array([1, 0]) - self.last_shot_position)  # Goal at (1,0) for right team
                    shooting_accuracy_reward[0] = max(0, 1 - distance)
                    self.shots_on_goal += 1
                    if observation['score'][1] > self.scored_goals:
                        # Reward more for scoring
                        shooting_accuracy_reward[0] += 1
                        self.scored_goals += 1
                self.last_shot_position = position.copy()
                
        reward[0] += shooting_accuracy_reward[0]
        components = {
            "base_score_reward": base_reward,
            "shooting_accuracy_reward": shooting_accuracy_reward
        }
        
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
