import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the synergistic effectiveness of midfield play by rewarding controlled transitions and pace management."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.team_possession_change = False
        self.previous_ball_owned_team = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.team_possession_change = False
        self.previous_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['team_possession_change'] = self.team_possession_change
        to_pickle['previous_ball_owned_team'] = self.previous_ball_owned_team
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.team_possession_change = from_pickle['team_possession_change']
        self.previous_ball_owned_team = from_pickle['previous_ball_owned_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward
        control_pace_reward = [0.0] * len(reward)
        transition_reward = [0.0] * len(reward)
        components = {"base_score_reward": base_score_reward, 
                      "control_pace_reward": control_pace_reward,
                      "transition_reward": transition_reward}

        if observation is None:
            return reward, components

        for i in range(len(observation)):
            o = observation[i]

            # Reward for maintaining ball control during transitions.
            if o['ball_owned_team'] == 0 and self.previous_ball_owned_team is not None \
               and self.previous_ball_owned_team != o['ball_owned_team']:
                transition_reward[i] += 0.5
                self.team_possession_change = True
            else:
                self.team_possession_change = False
            
            # Reward for controlling pace: less movement of the ball in the midfield decreases the pace.
            if o['ball_owned_team'] == 0 and abs(o['ball'][0]) <= 0.3:  # Checks if ball is in the midfield area.
                control_pace_reward[i] += 0.1 * (1.0 - np.linalg.norm(o['ball_direction']))

            # Compiling the final reward for the player
            reward[i] = base_score_reward[i] + control_pace_reward[i] + transition_reward[i]

        self.previous_ball_owned_team = observation[0]['ball_owned_team']

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
