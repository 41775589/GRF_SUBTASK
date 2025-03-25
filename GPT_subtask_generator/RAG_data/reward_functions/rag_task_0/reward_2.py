import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on offensive actions performed by agents.
    Specifically, it rewards accurate shooting, dribbling skill against opponents, and successful passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_counter = 0
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting": [0.0] * len(reward),
            "dribbling": [0.0] * len(reward),
            "passing": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            player_has_ball = (o['ball_owned_player'] == o['active']) and (o['ball_owned_team'] == 1)
            components['base_score_reward'][i] = reward[i]

            # Reward for shooting when close to the opponent's goal and controlling the ball
            distance_to_goal = np.linalg.norm([o['ball'][0] - 1, o['ball'][1]])
            if player_has_ball and distance_to_goal < 0.2:
                components['shooting'][i] = 1.0
                reward[i] += components['shooting'][i]

            # Reward for effective dribbling (player actively dribbling near opponents)
            if player_has_ball and 'action_dribble' in o['sticky_actions']:
                components['dribbling'][i] = 0.5
                self.dribbling_counter += 1
            reward[i] += self.dribbling_counter * components['dribbling'][i]

            # Reward for successful passes under pressure
            if player_has_ball and 'action_long_pass' in o['sticky_actions'] or 'action_high_pass' in o['sticky_actions']:
                components['passing'][i] = 0.3
                reward[i] += components['passing'][i]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
