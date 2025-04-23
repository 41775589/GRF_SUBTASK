import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds strategic defensive and counterattack rewards to teach agents defensive play and efficient transitions. """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.defensive_positions = np.zeros(10, dtype=int)  # Track defensive positioning
        self.counterattack_success = np.zeros(10, dtype=int)  # Track successful counterattacks
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions.fill(0)
        self.counterattack_success.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.defensive_positions.copy()
        to_pickle['counterattack_success'] = self.counterattack_success.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions = from_pickle['defensive_positions']
        self.counterattack_success = from_pickle['counterattack_success']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_position_reward": [0.0] * len(reward),
            "counterattack_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward defensive position: being close to their own goal when the opposite team has the ball
            if o['ball_owned_team'] == 1:  # If the opponent has the ball
                player_x = o['right_team'][o['active']][0]  # Player's x position
                if player_x < -0.5:  # Closer to own goal on the left
                    components["defensive_position_reward"][rew_index] = 0.05
                    self.defensive_positions[rew_index] += 1
            
            # Reward successful counterattacks: moving from own half to opponent's half quickly
            if o['ball_owned_team'] == 0:  # If own team has the ball
                transition_speed = np.linalg.norm(o['ball_direction'][:2])  # Speed of ball movement
                if transition_speed > 0.1 and o['ball'][0] > 0:  # Quickly moving towards opponent goal
                    components["counterattack_reward"][rew_index] = 0.1
                    self.counterattack_success[rew_index] += 1

            # Calculate total reward for this step
            reward[rew_index] += components["defensive_position_reward"][rew_index] + components["counterattack_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Add individual components to the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions for analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action  # Monitoring sticky actions frequency

        return observation, reward, done, info
