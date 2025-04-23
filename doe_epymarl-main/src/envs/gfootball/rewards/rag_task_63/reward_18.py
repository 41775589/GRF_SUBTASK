import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for training a goalkeeper in shot stopping, quick decision-making,
    and effective communication with defenders."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.shots_on_goal = 0  # Counter for shots on target
        self.goals_prevented = 0  # Counter for shots stopped or deflected
        self.quick_decision_making = 0  # Reward for quick decision-making executions
        self.communication_with_defenders = 0  # Reward for effective communication

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.shots_on_goal = 0
        self.goals_prevented = 0
        self.quick_decision_making = 0
        self.communication_with_defenders = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = super().get_state(to_pickle)
        state.update({
            "last_ball_position": self.last_ball_position,
            "shots_on_goal": self.shots_on_goal,
            "goals_prevented": self.goals_prevented,
            "quick_decision_making": self.quick_decision_making,
            "communication_with_defenders": self.communication_with_defenders
        })
        return state

    def set_state(self, state):
        from_pickle = super().set_state(state)
        self.last_ball_position = from_pickle["last_ball_position"]
        self.shots_on_goal = from_pickle["shots_on_goal"]
        self.goals_prevented = from_pickle["goals_prevented"]
        self.quick_decision_making = from_pickle["quick_decision_making"]
        self.communication_with_defenders = from_pickle["communication_with_defenders"]
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_reward = reward.copy()
        reward_details = {
            "base_score_reward": base_reward,
            "shot_stopping_reward": [0.0] * len(reward),
            "decision_making_reward": [0.0] * len(reward),
            "communication_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, reward_details

        # Detect shots on target
        if self.last_ball_position is not None and observation['ball'][0] > self.last_ball_position[0]:
            self.shots_on_goal += 1
            if observation['ball_owned_team'] == 0: # Left team is the opponent team
                reward_details['shot_stopping_reward'][0] = 1.0  # This would be the goalkeeper ag
                reward[0] += 1.0
                self.goals_prevented += 1

        # Assuming that making decisions quickly is beneficial:
        # We can make use of sticky actions to determine inactivity
        if not any(observation['sticky_actions']):
            self.quick_decision_making += 1
            reward_details['decision_making_reward'][0] = 0.3
            reward[0] += 0.3

        # For communication, a rough proxy could be number of defensive actions taken
        if (observation['game_mode'] >= 3) and (observation['game_mode'] <= 6):  # Assuming modes like free kick, corner etc are defensive-related
            self.communication_with_defenders += 1
            reward_details['communication_reward'][0] = 0.2
            reward[0] += 0.2

        self.last_ball_position = observation['ball'][:]
        return reward, reward_details
    
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
