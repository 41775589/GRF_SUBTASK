import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for collaborative plays between shooters (players in an
    advanced position ready to score) and passers (players that assists the setup of potential goals).
    This encourages players to make strategic passes that could lead to scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_to_shoot_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle['CheckpointRewardWrapper_previous_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "collaborative_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            current_ball_owner = o['ball_owned_player']

            if o['ball_owned_team'] == 1:  # If the ball is owned by the right team
                if self.previous_ball_owner is not None and self.previous_ball_owner != current_ball_owner:
                    # Finding pass relationships
                    if 'right_team_roles' in o:
                        roles = o['right_team_roles']
                        previous_roles = roles[self.previous_ball_owner]
                        current_roles = roles[current_ball_owner]
                        # Check if previous owner was a passer (midfielder) and the current is a shooter (forward)
                        if (previous_roles in {4, 5, 6, 7, 8} and current_roles in {9}):
                            components['collaborative_play_reward'][idx] += self.pass_to_shoot_reward
                            reward[idx] += components['collaborative_play_reward'][idx]

            self.previous_ball_owner = current_ball_owner
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
