import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards maintaining ball control under pressure and making strategic plays 
    to exploit open spaces and distribute the ball effectively.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "control_bonus": [0.0] * len(reward),
            "strategic_play_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Reward for maintaining ball control under pressure
        for idx, obs in enumerate(observation):
            if obs['ball_owned_team'] == 1 and obs['ball_owned_player'] == obs['active']:
                # Check if the team is under pressure (multiple opposing players nearby)
                team_pos = obs['left_team']
                ball_pos = obs['ball']
                distances = np.sqrt(((team_pos - ball_pos[:2])**2).sum(axis=1))
                pressure = np.sum(distances < 0.1)  # number of opponents within a certain distance

                if pressure > 2:  # more than two players nearby counts as "under pressure"
                    components["control_bonus"][idx] += 0.1  # reward for retaining ball under pressure

            # Reward for strategic play: exploit open spaces and effective ball distribution
            if obs['ball_owned_team'] == 1:  # Right team has the ball
                # Look for players in open space (no nearby opponents)
                for teammate_pos in obs['right_team']:
                    distances = np.sqrt(((obs['left_team'] - teammate_pos)**2).sum(axis=1))
                    if np.all(distances > 0.2):  # no opponents within a certain distance
                        components["strategic_play_bonus"][idx] += 0.05  # reward for being in open space

        # Updated rewards
        for idx in range(len(reward)):
            reward[idx] += (components["control_bonus"][idx] + components["strategic_play_bonus"][idx])

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
