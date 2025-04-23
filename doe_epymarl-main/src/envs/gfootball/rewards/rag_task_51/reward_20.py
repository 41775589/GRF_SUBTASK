import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances goalkeeper training with specific rewards related to shot-stopping, reflex actions, and
    initiating counter-attacks accurately.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shots_on_target = 0
        self.saves_made = 0
        self.passes_successfully_started = 0

        # Reward coefficients
        self.shot_stopping_reward = 1.0
        self.reflexes_reward = 0.5
        self.counter_attack_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shots_on_target = 0
        self.saves_made = 0
        self.passes_successfully_started = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shots_on_target'] = self.shots_on_target
        to_pickle['saves_made'] = self.saves_made
        to_pickle['passes_successfully_started'] = self.passes_successfully_started
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shots_on_target = from_pickle['shots_on_target']
        self.saves_made = from_pickle['saves_made']
        self.passes_successfully_started = from_pickle['passes_successfully_started']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "shot_stopping": [0.0] * len(reward),
                      "reflexes": [0.0] * len(reward),
                      "counter_attack": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goalie_role_index = np.where(o['left_team_roles'] == 0)[0][0]  # Assuming this agent is the left team goalie
            
            if o['ball_owned_team'] == 1 and o['active'] == goalie_role_index:  # ball is with opposing team
                distance = np.linalg.norm(o['left_team'][goalie_role_index] - o['ball'][0:2])
                if distance < 0.1:  # considers balls quite close as on target
                    self.shots_on_target += 1
                    components['shot_stopping'][rew_index] = self.shot_stopping_reward
            
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == goalie_role_index:
                if 'sticky_actions' in o and o['sticky_actions'][9]:  # dribble action by goalie
                    self.passes_successfully_started += 1
                    components['counter_attack'][rew_index] = self.counter_attack_reward

            # Reflex-based reward, assuming closer ball (when not owned) triggers a reflex necessity
            if o['ball_owned_team'] == 1:
                distance = np.linalg.norm(o['left_team'][goalie_role_index] - o['ball'][0:2])
                if distance < 0.07:
                    self.saves_made += 1
                    components['reflexes'][rew_index] = self.reflexes_reward
        
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
