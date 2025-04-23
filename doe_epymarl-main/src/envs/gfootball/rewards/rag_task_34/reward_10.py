import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards for decisive close-range attacks,
    focusing on dribbling and shooting precision near the opponent's goal."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.goal_zone_threshold = 0.2  # Define how close to the goal counts as the goal zone
        self.dribble_weight = 0.5  # Reward weight for effective dribbling
        self.shot_weight = 1.0  # Reward weight for shots on goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_and_shot_precision_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        # Assuming 2-agent setup; adjust if there are more agents.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][0]  # The X coordinate of the ball
            ball_owned_team = o['ball_owned_team']

            team_index = 0 if rew_index == 0 else 1
            own_goal = -1 if team_index == 0 else 1

            # Focus on events near the opponent’s goal
            if ball_position * own_goal > 1 - self.goal_zone_threshold:
                if ball_owned_team == team_index:
                    # Incrementally reward dribbling in the goal zone
                    if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                        components['dribble_and_shot_precision_reward'][rew_index] += self.dribble_weight
                    # Heavily reward shots taken in the goal zone towards the goal
                    if 'sticky_actions' in o and o['sticky_actions'][4] == 1:  # Assuming '4' is the shoot action index
                        components['dribble_and_shot_precision_reward'][rew_index] += self.shot_weight
            
            reward[rew_index] += components['dribble_and_shot_precision_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
