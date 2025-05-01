

OBJECTIVES_SUDOKU = ['number_adjacent_facts_other_value','number_other_facts_same_value',
                     'number_other_facts_other_value', 'adjacent_col_used', 'adjacent_row_used',
                     'adjacent_block_used', 'other_col_cons', 'other_row_cons', 'other_block_cons',
                     'number_adjacent_row_facts', 'number_adjacent_col_facts', 'number_adjacent_block_facts']

OBJECTIVES_LGP = ['adjacent_negative_facts','adjacent_facts_from_clue',
                  'adjacent_facts_from_bijectivity','adjacent_facts_from_transitivity',
                  'other_negative_facts','other_positive_facts','adjacent_clue','adjacent_bijectivity',
                  'adjacent_transitivity','other_clue','other_transitivity','other_bijectivity']


OBJECTIVE_NORMALIZED_SUDOKU = {'number_adjacent_facts_other_value':20,
                               'number_other_facts_same_value':8,
                               'number_other_facts_other_value':52,
                               'adjacent_col_used':1,
                               'adjacent_row_used':1,
                               'adjacent_block_used':1,
                               'other_col_cons':8,
                               'other_row_cons':8,
                               'other_block_cons':8,
                               'number_adjacent_row_facts':8,
                               'number_adjacent_col_facts':8,
                               'number_adjacent_block_facts':8}

LR_SUDOKU = {'baseline':0.5,
             'ucb':0.5,
             'disjunction':0.5,
             'coverage':1,
             'coverage_sum':0.5,
             'hamming':0.1,
             'w_disjunction':0.1,
             'w_coverage':5,
             'w_coverage_sum':0.5,
             'w_hamming':0.1,
             'lex_disjunction':0.1,
             'lex_coverage':0.1,
             'lex_coverage_sum':10,
             'lex_hamming':0.1,
             'cpucb_disjunction':0.1,
             'cpucb_coverage':10,
             'cpucb_coverage_sum':0.1,
             'cpucb_hamming':0.1
             }

LR_NORM_SUDOKU= {0:{'baseline':0.1,
                  'disjunction':0.1,
                  'hamming':0.5},
                1:{'baseline':0.1,
                  'disjunction':0.5,
                  'hamming':10},
                2:{'baseline':0.5,
                  'disjunction':10,
                   'w_disjunction':5,
                   'cpucb_disjunction':0.5,
                   'hamming':0.1,
                   'cpucb_hamming':5,
                   'w_hamming':1},
                3:{'baseline':0.5,
                  'disjunction':0.5,
                  'hamming':1}
                }

LR_LGP = {'baseline':5, #0.1
          'disjunction':5,  #5
          'hamming':50,  #5
          'w_disjunction':100,    #1
          'w_hamming':5,       #10
          'cpucb_disjunction':10,   #50
          'cpucb_hamming':10    #50
          }


LR_NORM_LGP = {0:{'baseline':0.1,
                  'disjunction':0.1,
                  'hamming':0.1},
               1:{'baseline':0.1,
                  'disjunction':0.5,
                  'hamming':1},
               2:{'baseline':0.1,
                  'disjunction':5,
                  'w_disjunction':0.1,
                  'cpucb_disjunction':0.5,
                  'hamming':10,
                  'cpucb_hamming':5,
                  'w_hamming':1},
               3:{'baseline':10,
                  'disjunction':10,
                  'hamming':0.5}
               }


#Computed as maximum
OBJECTIVES_NORMALIZED_LGP =     {'adjacent_negative_facts': 100,  #
                                'adjacent_facts_from_clue': 53,  # Correct, the might be all adj given the clues
                                'adjacent_facts_from_bijectivity': 24,  # Correct, checked in the code
                                'adjacent_facts_from_transitivity': 124,  # Correct, checked in the code
                                'other_negative_facts': 20,  # 150 - 20 for bijectivity - 20 x 3 for transitivity
                                'other_positive_facts': 5,  # 4~5 facts might be adjacent due to bijectivity
                                'adjacent_clue': 5,  # We do not know how many clues we have, suppose we have maximum 10 clues
                                'adjacent_bijectivity': 2,  # Correct, 1 for column and 1 for rows
                                'adjacent_transitivity': 6,  # Correct (2 times 3)
                                'other_clue': 8,  # Obtained as nadir point
                                'other_transitivity': 6,  # Correct
                                'other_bijectivity': 10}                        # Correct, 12 - 2 (1 for column and 1 for rows)



weights_custom = {'adjacent_negative_facts': 1.5,  #
                  'adjacent_facts_from_clue': 1.5,  # Correct, the might be all adj given the clues
                  'adjacent_facts_from_bijectivity': 1.5,  # Correct, checked in the code
                  'adjacent_facts_from_transitivity': 3.5,  # Correct, checked in the code
                  'other_negative_facts': 1.0,  # 150 - 20 for bijectivity - 20 x 3 for transitivity
                  'other_positive_facts': 1.5,  # 4~5 facts might be adjacent due to bijectivity
                  'adjacent_clue': 1.0,  # We do not know how many clues we have, suppose we have maximum 10 clues
                  'adjacent_bijectivity': 1.5,  # Correct, 1 for column and 1 for rows
                  'adjacent_transitivity': 2,  # Correct (2 times 3)
                  'other_clue': 1.5,  # Obtained as nadir point
                  'other_transitivity': 3,  # Correct
                  'other_bijectivity': 5}

WEIGHTED_DIVERSIFY = ['w_coverage', 'w_coverage_sum', 'w_disjunction']
LEX_DIVERSIFY = ['lex_coverage', 'lex_coverage_sum', 'lex_disjunction']
CPUCB_DIVERSIFY = ['cpucb_coverage', 'cpucb_coverage_sum', 'cpucb_disjunction']
