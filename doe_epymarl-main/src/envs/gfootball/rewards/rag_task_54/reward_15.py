import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for collaborative plays between shooters (players close to the goal)
    and passers (players setting up the play). The goal is to encourage strategic passing and shooting behavior
    to create and exploit scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.pass_bonus = 0.05
        self.shoot_bonus = 0.1
        self.shooting_threshold = 0.2  # Threshold for being considered close to the goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_bonus": [0.0] * len(reward),
                      "shoot_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'] if o['active'] < len(o['left_team']) else o['right_team'][o['active'] - len(o['left_team'])]
            ball_pos = o['ball'][:2]

            # Rewards for passing: Increase reward if a non-active player controls the ball and passes it forward
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] != o['active']:
                prev_pos = ball_pos
                new_ball_pos = ball_pos + o['ball_direction'][:2]
                if np.linalg.norm(new_ball_pos - prev_pos) > 0:
                    components['pass_bonus'][rew_index] += self.pass_bonus
                    reward[rew_index] += components['pass_bonus'][rew_index]

            # Rewards for shooting: Increase reward if the player is close to the goal and attempts a goal
            goal_pos = [1, 0] if o['ball_owned_team'] == 0 else [-1, 0]
            distance_to_goal = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))
            if distance_to_goal < self.shooting_threshold and o['sticky_actions'][9]:
                components['shoot_bonus'][rew_index] += self.shoot_bonus
                reward[rew_index] += components['shoot_bonus'][rew_index]

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
