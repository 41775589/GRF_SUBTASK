import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on enhancing the defending strategies by
    providing rewards for effective tackling, efficient movement to block opponents
    (stopping techniques), and making successful passes under pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialize tracking for sticky actions to encourage less frequent action change
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Parameters for additional rewards
        self.tackle_reward = 0.05
        self.positioning_reward = 0.03
        self.pressured_pass_reward = 0.07

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle["sticky_action_counts"] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get("sticky_action_counts", np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward),
                      "pressured_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            opponent_has_ball = o['ball_owned_team'] == 1
            player = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_pos = o['ball'][:2]

            # Reward tackling: when opponent owns the ball and ball changes possession near the player
            if opponent_has_ball and np.linalg.norm(ball_pos - player) < 0.05:
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]

            # Reward efficient movement (e.g., player positions himself between ball and goal effectively)
            goal_x = -1 if o['ball_owned_team'] == 1 else 1  # Goal position based on ball team possession
            vector_to_goal = np.array([goal_x, ball_pos[1]]) - player
            distance_to_goal_line = np.abs(player[0] - goal_x)
            if distance_to_goal_line < 0.1:
                alignment_score = -np.dot(vector_to_goal, np.array([-goal_x, 0]))
                if alignment_score > 0.1:  # player is aligning with the goal-line in Y axis properly
                    components["positioning_reward"][rew_index] = self.positioning_reward
                    reward[rew_index] += components["positioning_reward"][rew_index]

            # Reward successful passes under pressure
            # Assuming passing increases when multiple opponents are near
            opponents = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
            nearby_opponents = np.sum(np.linalg.norm(opponents - player, axis=1) < 0.2)
            if nearby_opponents > 2 and 'sticky_actions' in o and o['sticky_actions'][9]:  # assumes last action is a pass
                components["pressured_pass_reward"][rew_index] = self.pressured_pass_reward
                reward[rew_index] += components["pressured_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
