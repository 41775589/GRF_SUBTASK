from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = False
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.1, 0.0)

    # Setting up the left team with only necessary defenders and a goalkeeper.
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.8, -0.15, e_PlayerRole_CB)  # Centre Back left
    builder.AddPlayer(-0.8, 0.15, e_PlayerRole_CB)   # Centre Back right

    # Setting up the right team with attackers to test defensive strategies.
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.5, -0.2, e_PlayerRole_CF)  # Center Forward left
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CF)   # Center Forward right
    builder.AddPlayer(-0.6, 0.0, e_PlayerRole_AM)   # Attacking Midfielder center

    # This scenario sets the defense under test by having multiple attackers approach quickly.
