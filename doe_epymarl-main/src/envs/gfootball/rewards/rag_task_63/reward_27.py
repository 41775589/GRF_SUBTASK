import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_efficiency = 0.0
        self.defensive_efficiency = 0.0
        self.decision_making_efficiency = 0.0

    def reset(self):
        """
        Reset the environment and the performance tracking metrics.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_efficiency = 0.0
        self.defensive_efficiency = 0.0
        self.decision_making_efficiency = 0.0
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state and include custom state information.
        """
        to_pickle['goalkeeper_efficiency'] = self.goalkeeper_efficiency
        to_pickle['defensive_efficiency'] = self.defensive_efficiency
        to_pickle['decision_making_efficiency'] = self.decision_making_efficiency
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state and retrieve custom state information.
        """
        from_pickle = self.env.set_state(state)
        self.goalkeeper_efficiency = from_pickle.get('goalkeeper_efficiency', 0.0)
        self.defensive_efficiency = from_pickle.get('defensive_efficiency', 0.0)
        self.decision_making_efficiency = from_pickle.get('decision_making_efficiency', 0.0)
        return from_pickle

    def reward(self, reward):
        """
        Reward function structured to reinforce goalkeeper performance.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_efficiency": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            game_mode = o['game_mode']
            ball_owned_team = o['ball_owned_team']
            active_player_role = o['left_team_roles'][o['active']] if ball_owned_team == 0 else o['right_team_roles'][o['active']]
            
            # Reward goalkeeper based on successful saves (game_mode related to defense)
            if game_mode in {2, 3, 4} and active_player_role == 0:  # Game modes: GoalKick, FreeKick, Corner
                self.goalkeeper_efficiency += 0.2
                reward[rew_index] += self.goalkeeper_efficiency

            # Encourage quick decision making under pressure
            if ball_owned_team == 0 and active_player_role == 0:
                # Assume pressure is higher when the ball is close to the goal
                ball_y_position = abs(o['ball'][1])
                if ball_y_position < 0.2:
                    # Quick distribution rewards
                    self.decision_making_efficiency += 0.1
                    components["goalkeeper_efficiency"][rew_index] += self.decision_making_efficiency

            # Communication with defenders
            if ball_owned_team == 0 and active_player_role == 0:
                # Checking if defenders are in the correct position. Simplified by checking their y-position.
                defenders_positions = [player[1] for player in o['left_team'] if o['left_team_roles'][o['left_team'].index(player)] in {1, 2, 3}]  # CB, LB, RB roles.
                if all(-0.3 <= pos <= 0.3 for pos in defenders_positions):
                    self.defensive_efficiency += 0.3
                    reward[rew_index] += self.defensive_efficiency

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
