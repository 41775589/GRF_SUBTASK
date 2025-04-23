import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides a specialized reward for improvements in defending strategies such as tackling, efficient movement control, and pressured passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Include necessary state variables for pickling
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Initialize the observation from the environment.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "movement_control_reward": [0.0] * len(reward),
                      "pressured_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = None
            if o['ball_owned_team'] == 0:  # ball owned by left team
                active_player_pos = o['left_team'][o['active']]

            elif o['ball_owned_team'] == 1:  # ball owned by right team
                active_player_pos = o['right_team'][o['active']]

            if active_player_pos is not None:
                # Evaluating player movement control
                player_velocity = np.linalg.norm(o['left_team_direction'][o['active']]) if o['ball_owned_team'] == 0 else np.linalg.norm(o['right_team_direction'][o['active']])
                if player_velocity < 0.01:  # considering a very low speed as controlled movement
                    components['movement_control_reward'][rew_index] = 0.05 * (1 - player_velocity)
                    reward[rew_index] += components['movement_control_reward'][rew_index]

                # Evaluating pressured passing effectiveness
                if 'game_mode' in o and o['game_mode'] in {3, 5}:  # FreeKick or ThrowIn
                    components['pressured_pass_reward'][rew_index] = 0.1
                    reward[rew_index] += components['pressured_pass_reward'][rew_index]

                # Evaluating tackling
                if 'sticky_actions' in o:
                    is_tackling = o['sticky_actions'][0]  # hypothetical index for a tackling action
                    if is_tackling:
                        components['tackle_reward'][rew_index] = 0.2
                        reward[rew_index] += components['tackle_reward'][rew_index]

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
