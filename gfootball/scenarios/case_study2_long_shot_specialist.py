from . import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(-0.2, 0.0)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.2, 0.0, e_PlayerRole_CF)  # Long shooter with ball

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    # Two defenders blocking close approach
    builder.AddPlayer(-0.05, -0.05, e_PlayerRole_CB, controllable=False)
    builder.AddPlayer(-0.05, 0.05, e_PlayerRole_LB, controllable=False)