import gym
import numpy as np
class DefensiveSkillsRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance defensive skills and quick transitions
       for counter-attacks in a simulated football game environment."""

    def __init__(self, env):
        super(DefensiveSkillsRewardWrapper, self).__init__(env)
        self._collected_interceptions = {}
        self._interception_reward = 0.2
        self._quick_transition_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_interceptions = {}
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefensiveSkillsRewardWrapper'] = self._collected_interceptions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_interceptions = from_pickle['DefensiveSkillsRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for interception
            if ('ball_owned_team' in o and self.previous_ball_owner is not None and
                self.previous_ball_owner != o['ball_owned_team'] and
                o['ball_owned_team'] == o['designated']):
                # Only reward the player once for each interception during the game
                if self._collected_interceptions.get(rew_index, False) == False:
                    components["interception_reward"][rew_index] = self._interception_reward
                    reward[rew_index] += components["interception_reward"][rew_index]
                    self._collected_interceptions[rew_index] = True

            # Encourage quick transition for counter-attack
            # If the team just got the ball and is now in the opponent's half
            ball_position_x = o['ball'][0]
            if o['designated'] == o['active'] and ball_position_x > 0:
                if self.previous_ball_owner is not None and self.previous_ball_owner != o['ball_owned_team']:
                    components["transition_reward"][rew_index] = self._quick_transition_reward
                    reward[rew_index] += components["transition_reward"][rew_index]

            # Update previous ball owner
            self.previous_ball_owner = o['ball_owned_team']

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
