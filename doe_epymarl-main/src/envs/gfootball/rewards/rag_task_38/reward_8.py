import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for effectively transitioning from defense to attack
    through accurate long passes and quick transitions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_threshold = 0.8  # Threshold to consider a pass 'accurate'
        self.transition_speed_threshold = 0.5  # Maximum time steps for a quick transition
        self.reward_for_accurate_pass = 0.3
        self.reward_for_quick_transition = 0.5
        self.last_defensive_position = None
        self.transition_start_step = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_defensive_position = None
        self.transition_start_step = None
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "accurate_pass_reward": [0.0] * len(reward),
            "quick_transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check for effective defending phase ending with a pass
            if o['game_mode'] in [3, 6]:  # Assume game_mode 3 is free kick, 6 is a penalty
                self.last_defensive_position = np.array(o['ball'])
                self.transition_start_step = o['steps_left']

            # Tracking accuracy and quickness of the transition
            if o['ball_owned_team'] == 1:  # Assumes team 1 is the controlled team
                if self.last_defensive_position is not None and o['game_mode'] == 0:
                    distance_transitioned = np.linalg.norm(self.last_defensive_position - np.array(o['ball']))
                    steps_taken = self.transition_start_step - o['steps_left']

                    # Reward for long and accurate passes
                    if distance_transitioned > self.pass_accuracy_threshold:
                        components["accurate_pass_reward"][rew_index] = self.reward_for_accurate_pass
                        reward[rew_index] += components["accurate_pass_reward"][rew_index]

                    # Reward for quick transitions
                    if steps_taken <= self.transition_speed_threshold:
                        components["quick_transition_reward"][rew_index] = self.reward_for_quick_transition
                        reward[rew_index] += components["quick_transition_reward"][rew_index]

                    self.last_defensive_position = None
                    self.transition_start_step = None

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

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "last_defensive_position": self.last_defensive_position,
            "transition_start_step": self.transition_start_step
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.last_defensive_position = state_info['last_defensive_position']
        self.transition_start_step = state_info['transition_start_step']
        return from_pickle
