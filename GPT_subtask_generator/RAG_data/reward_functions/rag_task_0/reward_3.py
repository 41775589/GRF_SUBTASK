import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to foster offensive gameplay strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_reward = 0.1
        self.dribble_quality_reward = 0.1
        self.shoot_accuracy_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return state

    def set_state(self, state):
        state = self.env.set_state(state)
        self.sticky_actions_counter = np.array(state['sticky_actions_counter'])
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_quality_reward": [0.0] * len(reward),
            "dribble_quality_reward": [0.0] * len(reward),
            "shoot_accuracy_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            
            # Calculate distance to goal to estimate shooting potential
            goal_pos = [1, 0] if o['ball_owned_team'] == 0 else [-1, 0]
            distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(ball_pos))

            # Pass quality: reward successful passes that move closer to the opponent's goal
            if o['game_mode'] in [2, 3, 4, 5, 6]:  # Transition game modes like free kicks, throw-ins, etc.
                components['pass_quality_reward'][rew_index] = self.pass_quality_reward
                reward[rew_index] += components['pass_quality_reward'][rew_index]

            # Dribble quality: reward maintaining close control of the ball when near opponents
            if o['sticky_actions'][9] == 1 and np.linalg.norm(np.array(player_pos) - np.array(ball_pos)) < 0.05:
                components['dribble_quality_reward'][rew_index] = self.dribble_quality_reward
                reward[rew_index] += components['dribble_quality_reward'][rew_index]

            # Shooting accuracy: reward shots taken closer to the goal
            if o['game_mode'] == 6:  # Assuming mode 6 is a shooting scenario
                if distance_to_goal < 0.2:
                    components['shoot_accuracy_reward'][rew_index] = self.shoot_accuracy_reward
                    reward[rew_index] += components['shoot_accuracy_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
