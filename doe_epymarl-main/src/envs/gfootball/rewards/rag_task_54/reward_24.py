import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards effective collaboration between shooters and passers."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward scales for different collaborative actions
        self.pass_reward = 0.1
        self.shot_preparation_reward = 0.2
        self.successful_pass_reward = 0.3
        self.successful_shot_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_preparation_reward": [0.0] * len(reward),
                      "successful_pass_reward": [0.0] * len(reward),
                      "successful_shot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_has_ball = (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'])

            if player_has_ball:
                # Encourage passing behavior
                if self.sticky_actions_counter[9] > 0:  # action_dribble
                    components["pass_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += components["pass_reward"][rew_index]
                
                # Reward prepping for a shot
                if np.linalg.norm(o['ball_direction'][:2]) > 0.05:  # Assuming meaningful movement towards goal
                    components["shot_preparation_reward"][rew_index] = self.shot_preparation_reward
                    reward[rew_index] += components["shot_preparation_reward"][rew_index]

            # Reward successful passes (changes in ball possession among team members in proximity)
            if 'last_ball_owner' in o and o['last_ball_owner'] in o['left_team'] and o['ball_owned_player'] != o['last_ball_owner']:
                proximity = np.linalg.norm(o['left_team'][o['active']] - o['left_team'][o['last_ball_owner']])
                if proximity < 0.1:
                    components["successful_pass_reward"][rew_index] = self.successful_pass_reward
                    reward[rew_index] += components["successful_pass_reward"][rew_index]

            # Big reward for scoring
            if o['score'][0] > components['base_score_reward'][rew_index][0]:
                components["successful_shot_reward"][rew_index] = self.successful_shot_reward
                reward[rew_index] += components["successful_shot_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
