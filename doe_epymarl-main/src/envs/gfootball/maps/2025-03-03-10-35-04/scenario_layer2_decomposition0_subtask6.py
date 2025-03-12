from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Emphasizes continuity in play to practice passing under pressure

    builder.SetBallPosition(0.0, 0.0)

    # Setting up the player team with one midfielder who specializes in short passes
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper in the left team
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CM)  # Our agent playing as a Centre Midfielder

    # Configuring the opposing team for defensive pressure
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper in the right team
    builder.AddPlayer(0.1, 0.05, e_PlayerRole_CB)  # Opposing Centre Back close to our midfielder to apply pressure
    builder.AddPlayer(0.1, -0.05, e_PlayerRole_CB)  # Another opposing Centre Back to add more pressure
