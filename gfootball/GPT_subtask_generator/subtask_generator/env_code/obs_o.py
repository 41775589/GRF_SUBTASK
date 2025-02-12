{'right_team': array([[ 0.86112332,  0.04233703],
       [ 0.34571177,  0.04434711],
       [-0.14537288,  0.07615283],
       [ 0.39494497,  0.22443874],
       [ 0.46983981, -0.19208728]]), 'left_team': array([[-8.26035738e-01,  1.20851500e-05],
       [ 4.56890501e-02, -6.08141460e-02],
       [ 6.11223606e-03, -4.33273502e-02],
       [-5.31233028e-02, -6.94790483e-02],
       [-1.50302559e-01,  9.19511616e-02]]), 'right_team_roles': array([0, 7, 9, 2, 1]), 'score': [0, 0], 'left_team_yellow_card': array([False, False, False, False, False]), 'game_mode': 0, 'ball_direction': array([-0.00462898, -0.01706479, -0.04625521]), 'right_team_yellow_card': array([False, False, False, False, False]), 'left_team_roles': array([0, 7, 9, 2, 1]), 'ball_owned_player': -1, 'right_team_direction': array([[-2.43747933e-03, -4.84574353e-03],
       [-3.86960135e-04,  5.23936069e-05],
       [-4.22209705e-04, -2.19921960e-04],
       [-5.66664850e-03,  1.23156316e-03],
       [ 2.52863951e-03,  3.47246975e-03]]), 'ball': array([0.36277276, 0.11708356, 0.11968605]), 'left_team_tired_factor': array([0.00655901, 0.01070273, 0.00980258, 0.00703859, 0.0079844 ]), 'right_team_active': array([ True,  True,  True,  True,  True]), 'steps_left': 2951, 'ball_owned_team': -1, 'ball_rotation': array([ 1.73774140e-04, -2.42581882e-05, -2.29794765e-03]), 'left_team_direction': array([[ 0.        , -0.        ],
       [ 0.00070127,  0.00386324],
       [-0.00636069,  0.0006909 ],
       [-0.00116119,  0.00139133],
       [-0.00524205, -0.00341735]]), 'left_team_active': array([ True,  True,  True,  True,  True]), 'right_team_tired_factor': array([0.00743896, 0.01664084, 0.01563901, 0.01529056, 0.01886451]), 'designated': 2, 'active': 2, 'sticky_actions': array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=uint8)}


# List of observations exposed by the environment.
EXPOSED_OBSERVATIONS = frozenset({
    'ball', 'ball_direction', 'ball_rotation', 'ball_owned_team',
    'ball_owned_player', 'left_team', 'left_team_direction',
    'left_team_tired_factor', 'left_team_yellow_card', 'left_team_active',
    'left_team_roles', 'right_team', 'right_team_direction',
    'right_team_tired_factor', 'right_team_yellow_card', 'right_team_active',
    'right_team_roles', 'score', 'steps_left', 'game_mode'
})