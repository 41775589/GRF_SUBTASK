import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tactical reward focusing on midfield transitions and pace management."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_pace_control = 0.0
        self.midfield_transition_bonus = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # There are 10 possible sticky actions

    def reset(self):
        """ Resets the environment and the sticky action counter. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Get the state of the environment and add the current wrapper configuration. """
        to_pickle['CheckpointRewardWrapper'] = {'ball_pace_control': self.ball_pace_control, 
                                                'midfield_transition_bonus': self.midfield_transition_bonus}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set the state of the environment from pickle and extract wrapper configuration. """
        from_pickle = self.env.set_state(state)
        stored_state = from_pickle['CheckpointRewardWrapper']
        self.ball_pace_control = stored_state['ball_pace_control']
        self.midfield_transition_bonus = stored_state['midfield_transition_bonus']
        return from_pickle

    def reward(self, reward):
        """ Custom reward function focusing on midfield management and pace control. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pace_reward": np.zeros(len(reward)),
                      "transition_reward": np.zeros(len(reward))}

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            player_team_id = o['active']

            if player_team_id == -1:
                continue  # if there is no active player, skip this agent

            # Pace management reward
            current_ball_speed = np.linalg.norm(o['ball_direction'][:2])
            if current_ball_speed < 0.01:  # Threshold for low ball movement
                components["pace_reward"][i] = self.ball_pace_control
                reward[i] += components["pace_reward"][i]

            # Midfield transition reward
            if o['left_team'][player_team_id][0] > -0.2 and o['left_team'][player_team_id][0] < 0.2:
                components["transition_reward"][i] = self.midfield_transition_bonus
                reward[i] += components["transition_reward"][i]

        return reward, components

    def step(self, action):
        """ Step through environment and modify reward with custom rewards defined in reward method. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for j, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[j] += 1
                    info[f"sticky_actions_{j}"] = self.sticky_actions_counter[j]
        return observation, reward, done, info
