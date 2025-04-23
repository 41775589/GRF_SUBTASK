import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward function to focus on offensive strategies involving midfielders and strikers.
    Specifically, it rewards coordinated efforts between midfielders in creating spaces and delivering the ball, and
    strikers in finishing plays.
    """

    def __init__(self, env):
        super().__init__(env)
        self.midfielder_positions = []
        self.striker_positions = []
        self.ball_progress = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_positions = []
        self.striker_positions = []
        self.ball_progress = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'midfielder_positions': self.midfielder_positions,
            'striker_positions': self.striker_positions,
            'ball_progress': self.ball_progress
        }
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        state = self.env.set_state(from_pickle)
        data = from_pickle['CheckpointRewardWrapper']
        self.midfielder_positions = data['midfielder_positions']
        self.striker_positions = data['striker_positions']
        self.ball_progress = data['ball_progress']
        return state

    def reward(self, reward):
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_play_reward": [0.0] * len(reward)}

        if observations is None:
            return reward, components

        assert len(reward) == len(observations)

        for agent_index in range(len(reward)):
            obs = observations[agent_index]
            mid_positions = obs['left_team'][obs['left_team_roles'] == 4]  # Midfielders
            str_positions = obs['left_team'][obs['left_team_roles'] == 9]  # Strikers
            ball_position = obs['ball'][:2]

            # Track ball progress
            last_ball_pos = self.ball_progress.get(agent_index, (-1, -1))
            if np.linalg.norm(ball_position - last_ball_pos) > 0.01:  # if significant movement
                self.ball_progress[agent_index] = ball_position

                # Check if ball is progressing towards the striker from midfield
                midfield_to_ball = np.linalg.norm(mid_positions - ball_position, axis=1).min()
                ball_to_striker = np.linalg.norm(str_positions - ball_position, axis=1).min()

                if midfield_to_ball < 0.3 and ball_to_striker < 0.2:
                    # Reward this specific tactical play
                    components["positional_play_reward"][agent_index] = 0.5
                    reward[agent_index] += components["positional_play_reward"][agent_index]

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
