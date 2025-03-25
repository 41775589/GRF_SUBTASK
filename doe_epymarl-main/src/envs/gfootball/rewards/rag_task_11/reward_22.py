import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for enhancing offensive capabilities with precision and pace."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._goal_approach_reward = 0.03
        self._passed_to_advantage = 0.05
        self._successful_dribble = 0.02
        self._previous_ball_position = None
        self._opponent_goal_position = [1, 0]  # Assuming playing left to right

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['previous_ball_position'] = self._previous_ball_position
        return state

    def set_state(self, state):
        self._previous_ball_position = state['previous_ball_position']
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_approach_reward": [0.0] * len(reward),
                      "passed_to_advantage": [0.0] * len(reward),
                      "successful_dribble": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_position = o['ball'][:2]  # 2D position

            if self._previous_ball_position is not None:
                movement_towards_goal = np.dot(
                    np.array(self._opponent_goal_position) - np.array(self._previous_ball_position),
                    np.array(current_ball_position) - np.array(self._previous_ball_position)
                )
                if movement_towards_goal > 0:
                    components["goal_approach_reward"][rew_index] = self._goal_approach_reward
                    reward[rew_index] += components["goal_approach_reward"][rew_index]

            if (o['ball_owned_team'] == o['active'] and o['designated'] == o['active']):
                components["passed_to_advantage"][rew_index] = self._passed_to_advantage
                reward[rew_index] += components["passed_to_advantage"][rew_index]

            # Here, a successful dribble is implied if the ball is owned and player is sprinting
            # This could be improved with more context.
            if 'sticky_actions' in o and o['sticky_actions'][9]:  # dribble action active
                components["successful_dribble"][rew_index] = self._successful_dribble
                reward[rew_index] += components["successful_dribble"][rew_index]

            self._previous_ball_position = current_ball_position

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
