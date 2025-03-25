import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward to help train the agent for defensive abilities and quick transitions to counter-attacks."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for reward adjustments
        self.defensive_transition_reward = 0.2  # Reward for switching from losing possession to gaining possession
        self.defensive_action_reward = 0.1      # Reward for performing a defensive action
        self.opponent_approach_reward = 0.05    # Reward for reducing distance to the ball when the opponent has possession

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'defensive_transition_reward': [0.0] * len(reward),
                      'defensive_action_reward': [0.0] * len(reward),
                      'opponent_approach_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            prev_ball_owned_team = self.unwrapped.get_previous_observation()['ball_owned_team']

            # Trigger defensive transitions
            if prev_ball_owned_team == 1 and o['ball_owned_team'] == 0:  # From opponent to agent
                components['defensive_transition_reward'][rew_index] += self.defensive_transition_reward
                reward[rew_index] += components['defensive_transition_reward'][rew_index]

            # Encourage defensive actions if the opponent has the ball
            if o['ball_owned_team'] == 1:
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]  # Ignore z
                dist_to_ball = np.linalg.norm(player_pos - ball_pos)

                # Reward getting closer to the ball
                if dist_to_ball < self.last_distance_to_ball[rew_index]:
                    components['opponent_approach_reward'][rew_index] += self.opponent_approach_reward * (self.last_distance_to_ball[rew_index] - dist_to_ball)
                    reward[rew_index] += components['opponent_approach_reward'][rew_index]

                # Register the current distance for next step comparison
                self.last_distance_to_ball[rew_index] = dist_to_ball

            # Evaluate stick actions to encourage direct defensive maneuvers
            if o['sticky_actions'][0] or o['sticky_actions'][1] or o['sticky_actions'][2] or o['sticky_actions'][3]:
                components['defensive_action_reward'][rew_index] += self.defensive_action_reward
                reward[rew_index] += components['defensive_action_reward'][rew_index]

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
