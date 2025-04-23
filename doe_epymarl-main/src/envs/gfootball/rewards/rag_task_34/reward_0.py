import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents for effective dribbling and shooting close to the goal, 
    emphasizing quick decision-making and dribble effectiveness against goalkeepers."""

    def __init__(self, env):
        super().__init__(env)
        self.goal_position = np.array([1, 0])  # The goal position on the right side.
        self.distance_threshold = 0.2  # The threshold to consider close range.
        self.shot_reward = 1.0  # Reward for shooting under distance threshold.
        self.dribble_reward = 0.1  # Dribble effectiveness reward.
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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "close_range_shot_reward": [0.0] * len(reward),
                      "dribble_effectiveness_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Utilize observations to compute close-range shot and dribbling rewards.
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate distance to goal
            ball_position = np.array(o['ball'][:2])
            distance_to_goal = np.linalg.norm(ball_position - self.goal_position)
            
            # Check close-range and player has possession of the ball
            if distance_to_goal <= self.distance_threshold and o['ball_owned_team'] == 1:
                if o['ball_owned_player'] == o['active']:
                    # Provide rewards for shooting in close range
                    if 'action_shot' in o['sticky_actions']:
                        components["close_range_shot_reward"][rew_index] = self.shot_reward
                        reward[rew_index] += self.shot_reward
                    # Provide rewards for effective dribbling in close range
                    if 'action_dribble' in o['sticky_actions']:
                        components["dribble_effectiveness_reward"][rew_index] = self.dribble_reward
                        reward[rew_index] += self.dribble_reward

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
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
