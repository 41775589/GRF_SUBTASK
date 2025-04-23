import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for initiating counterattacks with long passes
    and quick transitions from defense to offense.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Constants to control the emphasis of different rewards
        self.counterattack_reward_coefficient = 5.0
        self.long_pass_reward_coefficient = 2.0
        self.transition_speed_coefficient = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = vars(self)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        for key, value in from_pickle.get('CheckpointRewardWrapper', {}).items():
            setattr(self, key, value)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counterattack_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "transition_speed_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Counterattack scenario: We look for a rapid change in possession from defense to attack
            if o['ball_owned_team'] == 1 and o['game_mode'] == 0: # Assuming team 1 is right team and is defending
                if 'ball_owned_player' in o and o['active'] == o['ball_owned_player']:
                    components["counterattack_reward"][rew_index] = self.counterattack_reward_coefficient

                    # Check if a long pass was made - large change in ball position
                    ball_pos_change = np.linalg.norm(o['ball_direction'][:2])
                    if ball_pos_change > 0.3:  # Threshold for considering a movement as a long pass
                        components["long_pass_reward"][rew_index] = self.long_pass_reward_coefficient

            # Speed of transition from defense to attack - uses ball speed as a proxy
            ball_speed = np.linalg.norm(o['ball_direction'][:2])
            components["transition_speed_reward"][rew_index] = self.transition_speed_coefficient * ball_speed

            # Reward to be actually assigned to an agent for this time step
            reward[rew_index] += components["counterattack_reward"][rew_index] \
                                 + components["long_pass_reward"][rew_index] \
                                 + components["transition_speed_reward"][rew_index]

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
