from . import *
def build_scenario(builder):
    builder.config().game_duration = 300
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.0, 0.0)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)

    # Our dribbling and sprint specialist player
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CF)

    # Opponent players simulate defense to penetrate
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.4, 0.05, e_PlayerRole_CB)
    builder.AddPlayer(-0.4, -0.05, e_PlayerRole_CB)
    builder.AddPlayer(-0.6, 0.1, e_PlayerRole_CM)
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CM)
