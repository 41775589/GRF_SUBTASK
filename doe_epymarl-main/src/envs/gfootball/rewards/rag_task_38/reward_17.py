import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that promotes training agents to specialize in counterattacks by
    rewarding successful defensive actions followed by rapid transitions to attack 
    with accurate long passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_defensive_pos = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_defensive_pos = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.last_defensive_pos
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_defensive_pos = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counterattack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            controlled_player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_pos = o['ball'][:2]

            # Identify defensive actions (winning back the ball in defensive half)
            if o['game_mode'] in {2, 3, 4} and np.abs(ball_pos[0]) > 0.5:  # Consider only actions in own half
                self.last_defensive_pos[rew_index] = controlled_player_pos

            # Reward moving into attack quickly after defense
            if 'ball_owned_team' in o and o['ball_owned_team'] != -1:
                old_pos = self.last_defensive_pos.get(rew_index)
                if old_pos is not None:
                    distance_moved = np.linalg.norm(np.array(controlled_player_pos) - np.array(old_pos))
                    # Reward long and forward passes from the last defensive position
                    if distance_moved > 0.5 and o['ball_owned_team'] == o['active']:  # Generic threshold for "long pass"
                        components["counterattack_reward"][rew_index] = 0.3  # Reward coefficient for successful transition
                        
                        # Further boost if it results in a goal opportunity or avoids defenders
                        if all(action == 0 for action in o['sticky_actions'][:6]) and o['sticky_actions'][9] == 1:
                            components["counterattack_reward"][rew_index] += 0.2

            if components["counterattack_reward"][rew_index] > 0:
                reward[rew_index] += components["counterattack_reward"][rew_index]

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
