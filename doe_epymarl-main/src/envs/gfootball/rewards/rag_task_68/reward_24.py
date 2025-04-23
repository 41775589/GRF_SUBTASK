import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for offensive strategies:
    - Shooting accuracy
    - Effective dribbling
    - Performing long and high passes
    This encourages learning a richer set of actions for offensive play.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.dribble_reward = 0.1
        self.pass_reward = 0.1
        self.shooting_accuracy_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        # Store needed states that are external to the environment state
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self.previous_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['CheckpointRewardWrapper']['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["dribble_reward"] = []
            components["pass_reward"] = []
            components["shooting_accuracy_reward"] = []

            # Add the dribbling reward
            if o['sticky_actions'][9] == 1:  # The dribble action index is 9
                components["dribble_reward"].append(self.dribble_reward)
                reward[rew_index] += self.dribble_reward
            
            # Evaluate shooting based on ball position & direction
            if self.previous_ball_position is not None and o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 1:
                ball_direction = o['ball_direction']
                ball_moved_towards_goal = (o['ball'][0] - self.previous_ball_position[0]) > 0
                if ball_moved_towards_goal and np.linalg.norm(ball_direction[:2]) > 1:
                    components["shooting_accuracy_reward"].append(self.shooting_accuracy_reward)
                    reward[rew_index] += self.shooting_accuracy_reward
            
            # Add pass reward on long or high passes, simulate checking the change in ball position
            if self.previous_ball_position is not None:
                distance_moved = np.linalg.norm(np.array(o['ball'][:2]) - np.array(self.previous_ball_position[:2]))
                if distance_moved > 0.5:  # assuming a threshold for a 'long pass'
                    components["pass_reward"].append(self.pass_reward)
                    reward[rew_index] += self.pass_reward

            # Update previous ball position
            self.previous_ball_position = o['ball']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            if isinstance(value, list):
                sum_value = sum(value)
            else:
                sum_value = value
            info[f"component_{key}"] = sum_value
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
