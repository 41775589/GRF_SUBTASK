import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive strategy by rewarding defensive positions and quick transitions to counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.past_ball_positions = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.past_ball_positions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        to_pickle['past_ball_positions'] = self.past_ball_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        self.past_ball_positions = from_pickle['past_ball_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        new_rewards = []
        reward_components = {'base_score_reward': reward.copy(), 'defensive_positioning': [], 'quick_transition': []}

        for player_index, obs in enumerate(observation):
            # Initialize rewards components for each action.
            defensive_reward = 0
            transition_reward = 0
            
            # Defensive positioning reward: higher when closer to own goal.
            own_goal_pos = -1 if obs['ball_owned_team'] == 1 else 1
            player_pos = obs['right_team' if obs['ball_owned_team'] == 1 else 'left_team'][obs['active']]
            distance_to_own_goal = abs(player_pos[0] - own_goal_pos)
            defensive_reward = (1 - distance_to_own_goal) * 0.2  # Normalizing factor

            # Quick transition reward: compare ball possession changes and movement towards the opponent goal.
            if len(self.past_ball_positions) > 1:
                last_ball_position = self.past_ball_positions[-1]
                current_ball_position = obs['ball'][:2]
                
                if obs['ball_owned_team'] == 1:  # Ball owned by the right team
                    if current_ball_position[0] > last_ball_position[0]:  # Ball is moving towards the left team's goal
                        transition_reward = 0.1

            reward_components['defensive_positioning'].append(defensive_reward)
            reward_components['quick_transition'].append(transition_reward)

            # Calculate final reward for this player
            total_reward = reward[player_index] + defensive_reward + transition_reward
            new_rewards.append(total_reward)

            # Update ball positions history.
            self.past_ball_positions.append(obs['ball'][:2])

        return new_rewards, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Aggregate components for logging
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info['component_' + key] = sum(value)
        
        # Update sticky actions counters
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info['sticky_actions_' + str(i)] = action

        return observation, reward, done, info
