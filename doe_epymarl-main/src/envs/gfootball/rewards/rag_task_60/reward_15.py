import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive adaptability."""

    def __init__(self, env):
        super().__init__(env)
        self.num_player_stops = None
        self.num_defensive_actions = None
        self.reset_counters()
        self. defensive_stop_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset_counters(self):
        self.num_player_stops = [0, 0]  # Resets stop counters for both agents
        self.num_defensive_actions = [0, 0]  # Resets defensive actions counters for both agents

    def reset(self):
        self.reset_counters()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'num_player_stops': self.num_player_stops,
            'num_defensive_actions': self.num_defensive_actions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self.num_player_stops = wrapper_state['num_player_stops']
        self.num_defensive_actions = wrapper_state['num_defensive_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for agent_index in range(len(reward)):
            o = observation[agent_index]

            # Logic to reward stops when opposition is close and moving fast towards the goal
            if 'right_team_direction' in o:
                player_speeds = np.linalg.norm(o['right_team_direction'], axis=1)
                player_positions = o['right_team'][:, 0]  # x-coordinates
                for speed, x_pos in zip(player_speeds, player_positions):
                    close_to_goal = x_pos > 0.5  # closer to the left team's goal
                    fast_approach = speed > 0.1

                    if close_to_goal and fast_approach and o['ball_owned_team'] == 1:
                        # Team 1 (right) owns the ball and is attacking fast near the goal
                        # Reward defensive stops or changes in directions which prevent goals
                        self.num_defensive_actions[agent_index] += 1
                        components['defensive_reward'][agent_index] = self.defensive_stop_reward
                        reward[agent_index] += 1.5 * self.defensive_stop_reward

            # Track changing from movement to stop
            previous_actions = self.sticky_actions_counter
            current_actions = o['sticky_actions']
            just_stopped = np.any((previous_actions[:2] == 1) & (current_actions[:2] == 0))

            if just_stopped:
                # Reward sudden stops if previously moving in the last step
                self.num_player_stops[agent_index] += 1
                components['defensive_reward'][agent_index] = self.defensive_stop_reward
                reward[agent_index] += 1 * self.defensive_stop_reward

            self.sticky_actions_counter = current_actions
            
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
