import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper that augments the reward mechanism to focus on specialized goalkeeper training.
    The reward function is designed to encourage:
    1. Shot-stopping: giving rewards when the goalkeeper stops incoming goals.
    2. Quick reflexes: rewarding fast changes in goalkeeper positioning in response to shot directions.
    3. Initiation of counter-attacks: giving rewards for accurate clearing passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters
        self.quick_reflex_reward = 0.1
        self.shot_stopping_reward = 0.5
        self.counter_attack_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Fetch observations to calculate additional rewards
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "quick_reflex_reward": [0.0] * len(reward),
            "shot_stopping_reward": [0.0] * len(reward),
            "counter_attack_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goalkeeper_idx = np.where(o['left_team_roles'] == 0)[0][0] # Assuming goalie is always in left team for simplicity.
            pos_goalie = o['left_team'][goalkeeper_idx]
            ball_position = o['ball'][:2]
            
            # Calculate distances
            distance_to_ball = np.linalg.norm(pos_goalie - ball_position)
            
            # Quick reflexes reward: Encourage minimizing the positional changes when the ball's moving fast
            if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
                components["quick_reflex_reward"][rew_index] = self.quick_reflex_reward / max(distance_to_ball, 0.1)

            # Shot-stopping reward: Reward if goalkeeper has the ball which was previously close to making a goal
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == goalkeeper_idx:
                if ball_position[0] < -0.7:  # Ball close to the left goal
                    components["shot_stopping_reward"][rew_index] = self.shot_stopping_reward

            # Counter attack initiation: Reward if ball is passed to a teammate far from the goal
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] != goalkeeper_idx:
                distance_from_goalie_to_player = np.linalg.norm(pos_goalie - o['left_team'][o['ball_owned_player']])
                if distance_from_goalie_to_player > 0.5:  # Ball passed to a teammate far from goalie
                    components["counter_attack_reward"][rew_index] = self.counter_attack_reward

            # Calculate final reward
            reward[rew_index] += (
                components["quick_reflex_reward"][rew_index]
                + components["shot_stopping_reward"][rew_index]
                + components["counter_attack_reward"][rew_index]
            )

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
