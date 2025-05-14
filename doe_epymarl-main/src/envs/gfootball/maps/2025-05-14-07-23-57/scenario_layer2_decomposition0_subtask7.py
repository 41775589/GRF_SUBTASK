from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000  # Extended duration for more complex defensive patterns
    builder.config().deterministic = False
    builder.config().offsides = True  # Enable offsides to mimic real match conditions and improve spatial awareness
    builder.config().end_episode_on_score = False
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Keep focus on player's defensive positioning

    # Set the ball near the opponent's midfield to simulate defensive line adjustment scenario (defensive half)
    builder.SetBallPosition(-0.3, 0)

    # Setting up the left team (training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper, not controllable
    # One player positioned as a Centre Back who is the focus of this training
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)
    
    # Setting up the right team (opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper, not controllable
    # Opponents positioned to challenge the trainee's defensive decision-making
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CF, lazy=True)  # Center Forward, lazy to restrict mobility - simulate pressing
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CF, lazy=True)  # Another Center Forward
    builder.AddPlayer(-0.3, 0.2, e_PlayerRole_CM, lazy=True)  # Midfielder on right
    builder.AddPlayer(-0.3, -0.2, e_PlayerRole_CM, lazy=True)  # Midfielder on left
