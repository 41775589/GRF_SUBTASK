from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.04,  0.040000, e_PlayerRole_RM)
  builder.AddPlayer(-0.04, -0.040000, e_PlayerRole_CF)
  builder.AddPlayer(-0.1, -0.1, e_PlayerRole_LB)
  builder.AddPlayer(-0.1,  0.1, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.04,  0.060000, e_PlayerRole_RM)
  builder.AddPlayer(-0.04, -0.060000, e_PlayerRole_CF)
  builder.AddPlayer(-0.1, -0.2, e_PlayerRole_LB)
  builder.AddPlayer(-0.1,  0.2, e_PlayerRole_CB)
