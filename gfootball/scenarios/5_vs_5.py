

from . import *

def build_scenario(builder):
  builder.config().game_duration = 3000
  builder.config().deterministic = False
  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
  builder.AddPlayer(0.000000,  0.020000, e_PlayerRole_RM)
  builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
  builder.AddPlayer(-0.1, -0.1, e_PlayerRole_LB)
  builder.AddPlayer(-0.1,  0.1, e_PlayerRole_CB)
  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
  builder.AddPlayer(-0.04,  0.040000, e_PlayerRole_RM)
  builder.AddPlayer(-0.04, -0.040000, e_PlayerRole_CF)
  builder.AddPlayer(-0.1, -0.1, e_PlayerRole_LB)
  builder.AddPlayer(-0.1,  0.1, e_PlayerRole_CB)
