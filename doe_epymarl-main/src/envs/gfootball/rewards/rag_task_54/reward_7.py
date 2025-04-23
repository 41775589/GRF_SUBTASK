import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense collaborative play reward emphasizing the interaction between shooters and passers."""

    def __init__(self, env):
        super().__init__(env)
        self._pass_to_shoot_transition = {}
        self.shooter_recognition_reward = 0.3
        self.pass_completion_reward = 0.2
        self.goal_bonus = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int) # Responsible for maintaining the count of sticky actions

    def reset(self):
        self._pass_to_shoot_transition = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._pass_to_shoot_transition
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._pass_to_shoot_transition = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "collaborative_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward joint actions which can be inferred from transitions and effective shooting.
            if o.get('ball_owned_team') == 0:  # Assuming team 0 is the controlled team
                ball_owner = o.get('ball_owned_player')
                previous_shooter = self._pass_to_shoot_transition.get(rew_index, {}).get('previous_shooter', None)

                if ball_owner != -1 and previous_shooter is not None and ball_owner == o['designated']:
                    components['collaborative_play_reward'][rew_index] = self.shooter_recognition_reward
                    reward[rew_index] += components['collaborative_play_reward'][rew_index]

                if previous_shooter is not None and 'score' in o:
                    if o['score'][0] > self._pass_to_shoot_transition.get(rew_index, {}).get('previous_score', [0, 0])[0]:
                        reward[rew_index] += self.goal_bonus

                self._pass_to_shoot_transition[rew_index] = {
                    'previous_shooter': ball_owner,
                    'previous_score': o['score']
                }

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
