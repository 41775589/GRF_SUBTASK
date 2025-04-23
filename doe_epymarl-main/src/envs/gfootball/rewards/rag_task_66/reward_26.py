import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for mastering short passes under defensive pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for all sticky actions
        self.passes_completed = 0  # Count of successful passes
        self.pass_accuracy_weight = 0.3  # Reward weight for pass accuracy

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'passes_completed': self.passes_completed}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passes_completed = from_pickle['CheckpointRewardWrapper']['passes_completed']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'pass_accuracy_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['game_mode'] in [0, 5]:  # Normal play or pass
                if ('ball_owned_team' in o and o['ball_owned_team'] == 0):  # Our team has the ball
                    if 'sticky_actions' in o and o['sticky_actions'][9]:  # Dribbling to maintain possession
                        self.sticky_actions_counter[9] += 1
                    if 'ball_owned_player' in o:
                        # Assuming function get_next_player_position() which derives the next player to receive the pass
                        next_player_position = self.get_next_player_position(o['ball_owned_player'],
                                                                             observation)
                        if next_player_position and np.linalg.norm(
                                np.subtract(o['ball'], next_player_position[:2])) < 0.1:
                            # Ball is close to the next player, which means a successful pass
                            self.passes_completed += 1
                            reward[rew_index] += self.pass_accuracy_weight
                            components['pass_accuracy_reward'][rew_index] = self.pass_accuracy_weight

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
                self.sticky_actions_counter[i] += int(action)
        return observation, reward, done, info

    def get_next_player_position(self, current_player, observation):
        # Dummy function to explain logic, replace or modify as per exact implementation needs
        return observation[0]['left_team'][current_player + 1] if current_player + 1 < len(observation[0]['left_team']) else None
