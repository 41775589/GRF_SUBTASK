from . import *
def build_scenario(builder):
    # Setting the game duration, determinism and the additional settings needed for the setup
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set an initial ball position
    builder.SetBallPosition(0.0, 0.0)

    # Setting the left team (training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CM, controllable=True)  # Sprinting player

    # Setting the right team (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper of the opposing team
    # Adding an opposing player to train against dynamic situations requiring sprint and stop
    builder.AddPlayer(-0.5, 0.1, e_PlayerRole_CM, controllable=False)
