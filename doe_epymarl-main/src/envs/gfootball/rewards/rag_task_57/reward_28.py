import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on coordinated offensive plays 
    between midfielders and strikers.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfielder_coordination_bonus = 0.1
        self.striker_finish_bonus = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfielder_coordination": [0.0] * len(reward),
                      "striker_finish": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            active_player_role = o['right_team_roles'][o['active']] if o['ball_owned_team'] == 1 else o['left_team_roles'][o['active']]

            # Reward for midfielder coordination
            if active_player_role in [4, 5, 6]:  # Assuming roles 4,5,6 are midfielders
                controlled_by_midfielder = o['sticky_actions'][9]  # Assuming index 9 is coordination-related action.
                if controlled_by_midfielder:
                    components["midfielder_coordination"][i] = self.midfielder_coordination_bonus
                    reward[i] += components["midfielder_coordination"][i]

            # Reward for striker finish
            if active_player_role in [9]:  # Assuming role 9 is striker
                near_opponent_goal = (o['ball'][0] > 0.5)  # Assuming the goal y-range
                if near_opponent_goal and o['ball_owned_team'] == 1:
                    components["striker_finish"][i] = self.striker_finish_bonus
                    reward[i] += components["striker_finish"][i]

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
