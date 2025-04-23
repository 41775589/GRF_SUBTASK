import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on improving defensive capabilities, 
    targeting goalkeeper shot-stopping and defenders' tackling and ball retention."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, dtype=np.float)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]

            active_player_role = o['right_team_roles'][o['active']]
            ball_owner_role = o['right_team_roles'][o.get('ball_owned_player', -1)]
            ball_owner_team = o['ball_owned_team']
            ball_distance = np.linalg.norm(o['ball'][:2])

            # Encourage goalkeepers (role '0' denotes goalkeeper)
            if active_player_role == 0 and ball_owner_role == 0 and ball_owner_team == 1:
                # Reward successful shot stopping and clearances by the goalkeeper
                components.setdefault('goalkeeper_reward', np.zeros(len(reward)))
                components['goalkeeper_reward'][idx] = 0.2  # additional reward for goalkeeper actions
                reward[idx] += components['goalkeeper_reward'][idx]

            # Encourage defenders
            if active_player_role in [1, 2, 3, 4] and ball_owner_team == 1:
                # Reward tackling and ball retention for defenders
                components.setdefault('defender_reward', np.zeros(len(reward)))
                components['defender_reward'][idx] = 0.1  # additional reward for defending actions
                reward[idx] += components['defender_reward'][idx]

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
