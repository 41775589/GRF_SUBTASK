import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specialized reward scheme for goalkeeper training.
    It focuses on blocking shots (shot-stopping), improving reflexes (via position adjustment),
    and initiating counter-attacks through accurate passes by the goalkeeper.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_pos = None

    def reset(self):
        self.last_ball_pos = None
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_last_ball_pos'] = self.last_ball_pos
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_pos = from_pickle.get('CheckpointRewardWrapper_last_ball_pos', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward}
        
        components = {
            "base_score_reward": reward.copy(),
            "shot_stopping_reward": [0.0] * len(reward),
            "reflex_improvement_reward": [0.0] * len(reward),
            "counter_attack_reward": [0.0] * len(reward)
        }
        
        # Implemented rewards for goalie training:
        for rew_index, o in enumerate(observation):
            current_ball_pos = o['ball']

            # Reward for shot stopping
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                if current_ball_pos[0] > 0.7:  # Ball is close to own goal
                    components['shot_stopping_reward'][rew_index] += 5.0
                    reward[rew_index] += components['shot_stopping_reward'][rew_index]

            # Reward for reflex improvement
            if self.last_ball_pos is not None:
                speed = np.linalg.norm(current_ball_pos - self.last_ball_pos[:2])
                if speed > 0.01:  # Ball is moving significantly
                    components['reflex_improvement_reward'][rew_index] += speed * 2.0
                    reward[rew_index] += components['reflex_improvement_reward'][rew_index]

            # Reward for initiating counter-attacks
            if (o['ball_owned_team'] == 0 and o['active'] == 0 and  # Ball owned by goalkeeper
                o['ball_direction'][0] > 0.1):  # Ball is played forward emphatically
                components['counter_attack_reward'][rew_index] += 2.0
                reward[rew_index] += components['counter_attack_reward'][rew_index]

        self.last_ball_pos = np.array(observation[0]['ball'])  # Update last ball position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions information
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
