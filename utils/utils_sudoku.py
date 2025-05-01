from time import time

from cpmpy import intvar
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl

from utils.constants import WEIGHTED_DIVERSIFY, LEX_DIVERSIFY, CPUCB_DIVERSIFY, OBJECTIVES_SUDOKU
from utils.utils import split_ocus_assum, leftover, rank_dictionary, ucb_dictionary, compute_normalization, \
    create_folders
from utils.utils_classes import HintType


def load_sudokus_from_json(filename="./data/sudoku_instances.json"):
    # Load all Sudoku puzzles from a JSON file as an array
    with open(filename, 'r') as file:
        sudokus_list = json.load(file)
    sudokus = [np.array(lst) for lst in sudokus_list]
    return sudokus




def generate_steps_sudoku(grid, sequential_sudoku=-1, feature_weights=None, counter=None, diversify=False,
                          obj_values_gt=None, to_return=False, gamma=0, no_user=None, is_tqdm=True,
                          normalization=[],fact_to_exp=None, is_random= False,trade_offs=None,
                          iterations=None):
    if feature_weights==None:
        no_user = 'SMUS'
        feature_weights = {
            'number_facts': 1,
            'number_constraints': 1,
        }
    if is_random:
        no_user = 'random'
    if counter == None:
        counter = len(grid[grid == 0])
    info = []
    if is_tqdm:
        to_consider = tqdm(range(counter))
    else:
        to_consider = range(counter)
    for _ in to_consider:
        generator = get_split_ocus_hint_sudoku(given=grid, solvername="exact", param_dict=dict(),
                                               feature_weights=feature_weights, hs_solver_name="gurobi",
                                               diversify=diversify, obj_values_gt=obj_values_gt, gamma=gamma,
                                               normalization=normalization,
                                               fact_to_exp=fact_to_exp,is_random=is_random,trade_offs=trade_offs,
                                               iterations=iterations)
        time_start = time()
        expl, constraints_expl,features = next(generator)
        time_end = time()
        if sequential_sudoku >= 0:
            row, col, val = expl["hint_derived"][1]
            grid[row, col] = val
            expl['time'] = time_end - time_start
            info.append(expl)
            df = pd.DataFrame(info)
            if not to_return:
                create_folders(f"./data/gt_sudoku/")
                df.to_csv(f"./data/gt_sudoku/sudoku_user_{no_user}_sudoku_{sequential_sudoku}.csv", index=False)
        else:
            return expl, constraints_expl, features
    return df


def generate_top_k_expl(grid,df_steps_file,index,top_k=100,feature_weights=None,timeout_top_k=36000,type_gen='SMUS'):
    df_steps = pd.read_csv(df_steps_file)
    all_explanation_steps = []
    n_to_expl = len(grid[grid == 0])
    if feature_weights is None:
        feature_weights = {
            'number_facts': 1,
            'number_constraints': 1,
        }
    start = time()
    for pos_in_seq in range(n_to_expl):
        to_be_explained = df_steps.iloc[pos_in_seq].tolist()
        generator = get_split_ocus_hint_sudoku(given=grid.copy(), solvername="exact", param_dict=dict(),
                                               feature_weights=feature_weights,
                                               time_limit=leftover(timeout_top_k,start),
                                               hs_solver_name="gurobi", single=True,
                                               row_to_explain=to_be_explained[0],
                                               col_to_explain=to_be_explained[1])
        for alt_cnt in tqdm(range(top_k)):
            hint_k = next(generator)
            if hint_k is not None:
                hint_derived = hint_k["hint_derived"]
                all_explanation_steps.append(hint_k)
                df = pd.DataFrame(all_explanation_steps)
                df.to_csv(f'./data/top_k/sudoku_{index}_{type_gen}.csv', index=False)
                if alt_cnt == 0:
                    _, (row, col, val) = hint_derived
                if leftover(timeout_top_k,start)==0:
                    break
            else:
                break
        grid[row, col] = val






def get_split_ocus_hint_sudoku(given: np.ndarray, solvername="exact", param_dict=dict(), feature_weights=dict(),
                               time_limit=3600, hs_solver_name="gurobi",diversify=False, obj_values_gt=None, gamma=0,
                               normalization=[],fact_to_exp=None,is_random=False,trade_offs=None,
                               iterations=None):
    #definition weights_exploration
    weights_exploration = {}
    if diversify in WEIGHTED_DIVERSIFY or diversify=='w_hamming':
        for obj in feature_weights:
            weights_exploration[obj] = feature_weights[obj]


    elif diversify in LEX_DIVERSIFY or diversify=='lex_hamming':
        ranked_trade_offs = rank_dictionary(trade_offs)
        for obj in feature_weights:
            weights_exploration[obj] = ranked_trade_offs[obj]


    if diversify== 'ucb':
        ranked_trade_offs = ucb_dictionary(trade_offs.copy(),feature_weights,iterations)
        for obj in feature_weights:
            weights_exploration[obj] = 0
            feature_weights[obj] = ranked_trade_offs[obj]


    elif diversify in CPUCB_DIVERSIFY or diversify== 'cpucb_hamming':
            # max_v = max(feature_weights.values())
            # feature_weights = {k: v / max_v for k, v in feature_weights.items()}
            ranked_trade_offs = ucb_dictionary(trade_offs,feature_weights,iterations)
            for obj in feature_weights:
                weights_exploration[obj] = ranked_trade_offs[obj]


    n_rows = n_cols = n_vals  = given.shape[0]

    # boolean variables for each integer var in sudoku
    bvars = []
    map_bvar_pos_val = {}
    for r in range(n_rows):
        for c in range(n_cols):
            bv_var_vals = []
            for val in range(n_vals):
                bv = cp.boolvar(name=f"given[{r},{c}]={val + 1}")
                map_bvar_pos_val[bv] = (r, c, val + 1)
                bv_var_vals.append(bv)
            bvars.append(bv_var_vals)

    bvars = cp.cpm_array(bvars)
    ### values can appear only once at every position in the sudoku grid
    mapping_cons = [cp.sum(bvs) == 1 for bvs in bvars]

    ### ---- CONSTRAINTS
    constraints, row_map, col_map, block_map = get_sudoku_constraints(bvars)
    rev_row_map = {tuple(v): k for k, v in row_map.items()}
    rev_col_map = {tuple(v): k for k, v in col_map.items()}
    rev_block_map = {(v[0], tuple(tuple(inner) for inner in v[1])): k for k, v in block_map.items()}

    ### ---- FACTS
    facts = get_sudoku_facts(bvars, given)
    assert cp.Model(constraints + facts + mapping_cons).solveAll(solution_limit=2) == 1, "Found more than 1 solution, or model is UNSAT!!"

    ### ---- TO Explain
    if fact_to_exp is not None:
        # to_explain_for_norm =  get_facts_to_explain(bvars, given, single=False,is_random=is_random)
        to_explain = get_facts_to_explain(bvars, given, single=True,single_row=fact_to_exp[0],single_col=fact_to_exp[1],is_random=is_random)
    else:
        # to_explain_for_norm = get_facts_to_explain(bvars, given, single=False, is_random=is_random)
        to_explain = get_facts_to_explain(bvars, given, single=False,is_random=is_random)
    ### ---- SOFT Constraints
    soft = facts + constraints + to_explain

    assump = cp.boolvar(shape=len(soft), name="assumption")


    for a_var, soft_con in zip(assump, soft):
        if soft_con in row_map:
            _, row_idx = row_map[soft_con]
            a_var.name = f"->row-{row_idx}"
        elif soft_con in col_map:
            _, col_idx = col_map[soft_con]
            a_var.name = f"->col-{col_idx}"
        elif soft_con in block_map:
            _, ((i1, j1), (i2, j2)) = block_map[soft_con]
            a_var.name = f"->block-[{i1}:{i2}, {j1}:{j2}]"
        else:
            a_var.name = "->" + str(soft_con)


    assum_map = dict(zip(assump, soft))
    con_map = dict(zip(soft, assump))

    oneof_idxes = np.arange(len(facts + constraints), len(facts + constraints + to_explain))


    ## Reify assumption variables
    hard = mapping_cons + [assump.implies(soft)]  # each assumption variable implies a candidate
    assert bvars.shape == (n_rows * n_cols, n_vals), f"Shape of bvars should be (81,9) but got {bvars.shape}"


    objectives = []
    all_bv_blocks = set(con_map[c] for c in set(block_map))
    all_bv_rows = set(con_map[c] for c in set(row_map))
    all_bv_cols =  set(con_map[c] for c in set(col_map))
    all_bv_constraints = all_bv_blocks | all_bv_rows | all_bv_cols
    all_bv_facts = set(assump[:len(facts)])

    ## Definition features for each explained fact
    map_explained_features = {}
    tmp_explained_features = {}
    hs_hard_explained = {}
    if not diversify:
        obj_values_gt = None

    ##Definition normalization
    # for bv_explained in to_explain_for_norm:
    #     feature_obj_map,feature_obj_map_norm = map_sudoku_features(given, bvars, bv_explained, all_bv_constraints, all_bv_facts,
    #                                                                all_bv_rows, all_bv_cols, all_bv_blocks, con_map,
    #                                                                rev_row_map, rev_col_map, rev_block_map, map_bvar_pos_val)
    #     tmp_explained_features[bv_explained] = feature_obj_map_norm

    # objective_normalized_sudoku = compute_normalization(tmp_explained_features,normalization,problem_type='sudoku')
    objective_normalized_sudoku = normalization
    for bv_explained in to_explain:
        feature_obj_map,_ = map_sudoku_features(given, bvars, bv_explained, all_bv_constraints, all_bv_facts,
                                                all_bv_rows, all_bv_cols, all_bv_blocks, con_map,
                                                rev_row_map, rev_col_map, rev_block_map, map_bvar_pos_val)

        map_explained_features[con_map[bv_explained]] = feature_obj_map

        objective = 0
        tmp = 0
        all_feature_weights = []
        all_used_features = []
        normalized_weights = {}
        integer_normalization = {}
        for feature, feature_weight in feature_weights.items():
            all_feature_weights.append(feature_weight)
            all_used_features.append(feature_obj_map[feature])
            if diversify== 'baseline' or diversify== 'disjunction' or diversify== 'coverage' or diversify== 'coverage_sum':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_sudoku[feature])
                integer_normalization[feature] = int(1e5 / objective_normalized_sudoku[feature])
                tmp += feature_obj_map[feature] * normalized_weights[feature]
            elif diversify in WEIGHTED_DIVERSIFY or diversify in LEX_DIVERSIFY or diversify in CPUCB_DIVERSIFY or diversify=='ucb':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_sudoku[feature])
                integer_normalization[feature] = int(1e5 * weights_exploration[feature] / objective_normalized_sudoku[feature])
                tmp += feature_obj_map[feature] * normalized_weights[feature]
            elif diversify == 'hamming':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_sudoku[feature])
                integer_normalization[feature] = int(1e5)
                tmp += feature_obj_map[feature] * normalized_weights[feature]
            elif diversify == 'w_hamming' or diversify == 'lex_hamming' or diversify == 'cpucb_hamming':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_sudoku[feature])
                integer_normalization[feature] = int(1e5 * weights_exploration[feature])
                tmp += feature_obj_map[feature] * normalized_weights[feature]
            else: #no diversification
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_sudoku[feature])
                tmp += feature_obj_map[feature] * normalized_weights[feature]
        if gamma is None:
            objective += tmp
            gamma = 1
        else:
            objective += tmp * (1 - gamma)
        if diversify == 'baseline':
            different = sum(cp.max(integer_normalization[feature]*(feature_obj_map[feature] - value),
                                   integer_normalization[feature]*(value - feature_obj_map[feature])) for feature, value in obj_values_gt.items())
            objective += -gamma * (different)
            temp_list = [map_explained_features[con_map[bv_explained]][feature] != obj_values_gt[feature] for feature in obj_values_gt]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'ucb':
            temp_list = [
                (map_explained_features[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'disjunction' or diversify == 'w_disjunction' or diversify == 'cpucb_disjunction' or diversify == 'lex_disjunction':
            different = sum(cp.max(integer_normalization[feature] * (feature_obj_map[feature] - value),
                                   integer_normalization[feature] * (value - feature_obj_map[feature])) for
                            feature, value in obj_values_gt.items())
            objective += -gamma * (different)
            temp_list = [
                (map_explained_features[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'coverage' or diversify == 'w_coverage' or diversify == 'cpucb_coverage' or diversify == 'lex_coverage':
            different = cp.max(integer_normalization[feature] * (value - feature_obj_map[feature]) for feature, value in
                               obj_values_gt.items())
            objective += - gamma * different
            temp_list = [
                (map_explained_features[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'coverage_sum' or diversify == 'w_coverage_sum' or diversify == 'cpucb_coverage_sum' or diversify == 'lex_coverage_sum':
            different = sum(integer_normalization[feature] * (value - feature_obj_map[feature]) for feature, value in
                            obj_values_gt.items())
            objective += - gamma * different
            temp_list = [
                (map_explained_features[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'hamming' or diversify == 'w_hamming' or diversify == 'cpucb_hamming' or diversify == 'lex_hamming':
            different = sum(integer_normalization[feature] * (cp.sum([feature_obj_map[feature]]) != value) for feature, value in obj_values_gt.items())
            objective += - gamma * different
            # temp_list = [
            #     (map_explained_features[con_map[bv_explained]][feature] < obj_values_gt[feature])
            #     if feature_weights[feature] >= 0
            #     else (map_explained_features[con_map[bv_explained]][feature] > obj_values_gt[feature])
            #     for feature in obj_values_gt
            # ]
            # hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)


        objective += con_map[bv_explained]
        objectives.append(objective)

    gen = split_ocus_assum(soft=assump, oneof_idxes=oneof_idxes, dmap=assum_map,
                           objectives=objectives, hard=hard, solver=solvername,
                           hs_solver_name=hs_solver_name, solver_params=param_dict,
                           time_limit=time_limit,hard_hs=hs_hard_explained)

    for expl_cons in gen:
        explained = [~bv for bv in expl_cons if bv in set(to_explain)]
        assert len(explained) == 1, f"{explained=}"

        features_map =  map_explained_features[con_map[~explained[0]]]
        featurized_expl = featurize_expl_sudoku(expl_cons,features_map,con_map)


        explained_fact = get_integer_facts(bvars, explained, HintType.DERIVED.name)

        used_facts = get_integer_facts(bvars, expl_cons)
        hint_constraints = ([row_map[cons] for cons in expl_cons if cons in row_map] +
                            [col_map[cons] for cons in expl_cons if cons in col_map] +
                            [block_map[cons] for cons in expl_cons if cons in block_map])

        hint = {
            'grid': given.copy(),
            "hint_facts": used_facts,
            "hint_constraints": hint_constraints,
            "hint_derived": explained_fact[0]
        }

        hint.update(featurized_expl)

        yield hint,expl_cons,featurized_expl

def get_sudoku_constraints(bvars):
    """
        transform classic sudoku constraints into constraints over boolean variables
    """
    _, nvals = bvars.shape
    block_offset = int(nvals **(1/2))
    assert bvars.shape == (nvals*nvals, nvals), f"Shape of bvars should be {(nvals*nvals, nvals)} but got {bvars.shape}"
    bvars = bvars.reshape((nvals, nvals, nvals))  # 3D cube with boolean vars

    constraints = []

    # alldiff row constraints
    row_map = dict()
    for idx,row in enumerate(bvars):
        cons = cp.all([cp.sum(row[:,val]) <= 1 for val in range(nvals)])
        constraints.append(cons)
        row_map[cons] = [HintType.ROW.name, idx]

    # alldiff col constraints
    col_map = dict()
    for idx, col in enumerate(np.transpose(bvars, axes=[1, 0, 2])):
        cons = cp.all([cp.sum(col[:, val]) <= 1 for val in range(nvals)])
        constraints.append(cons)
        col_map[cons] = [HintType.COL.name, idx]

    # alldiff block constraints
    block_map = dict()
    for i in range(0, nvals, block_offset):
        for j in range(0, nvals, block_offset):
            block = bvars[i:i+block_offset, j:j+block_offset].reshape((nvals, nvals))
            cons = cp.all([cp.sum(block[:, val]) <= 1 for val in range(nvals)])
            constraints.append(cons)
            block_map[cons] = [HintType.BLOCK.name, [[i, j], [i+block_offset, j+block_offset]]]

    return constraints, row_map, col_map, block_map

def get_sudoku_facts(bvars, given):
    """
        transform given digits into constraint over boolean variables
    """
    (nrows, ncols) = given.shape
    assert bvars.shape == (nrows * ncols, nrows), f"Shape of bvars should be ({nrows * ncols}, {nrows}) but got {bvars.shape}"
    temp_bvars = bvars.reshape((nrows, ncols, nrows))

    facts = []
    for row in range(nrows):
        for col in range(ncols):
            if given[row, col] != 0:
                facts.append(temp_bvars[row, col, given[row, col] - 1])
    return facts


def get_facts_to_explain(bvars, given,single=False,single_row=None, single_col=None,is_random=False):
    """
        get the negated boolean variables corresponding to values in the solution of non-filled squares
    """
    (nrows, ncols) = given.shape
    assert bvars.shape == (nrows * ncols, nrows), f"Shape of bvars should be ({nrows * ncols}, {nrows}) but got {bvars.shape}"
    bvars = bvars.reshape((nrows, ncols, nrows))

    to_explain = []
    if single:
        to_explain += [~bv for bv in bvars[single_row,single_col] if bv.value() is True]
    elif is_random:
        for row in range(nrows):
            for col in range(ncols):
                if given[row, col] == 0:
                    to_explain += [~bv for bv in bvars[row, col] if bv.value() is True]
        to_explain = [np.random.choice(to_explain)]
    else:
        for row in range(nrows):
            for col in range(ncols):
                if given[row, col] == 0:
                    to_explain += [~bv for bv in bvars[row, col] if bv.value() is True]
    return to_explain


def map_sudoku_features(given,bvars,bv_explained,all_bv_constraints,all_bv_facts,
                        all_bv_rows,all_bv_cols,all_bv_blocks,con_map,rev_row_map,rev_col_map,rev_block_map,
                        map_bvar_pos_val):
    _, nvals = bvars.shape
    block_offset = int(nvals ** (1 / 2))

    ### Initialising the features of explanations

    (explained_row, explained_col, value_explained) = map_bvar_pos_val[~bv_explained]
    explained_block_row = (explained_row //block_offset)*block_offset
    explained_block_col = (explained_col //block_offset)*block_offset

    bv_adjacent_row_cons = con_map[rev_row_map[(HintType.ROW.name, explained_row)]]
    bv_adjacent_col_cons = con_map[rev_col_map[(HintType.COL.name, explained_col)]]
    bv_adjacent_block_cons = con_map[rev_block_map[
            (HintType.BLOCK.name, ((explained_block_row, explained_block_col), (explained_block_row+block_offset, explained_block_col+block_offset)))
    ]]

    bv_other_row_cons = all_bv_rows - set({bv_adjacent_row_cons})
    bv_other_col_cons = all_bv_cols - set({bv_adjacent_col_cons})
    bv_other_block_cons = all_bv_blocks - set({bv_adjacent_block_cons})

    adjacent_sudoku = np.zeros(given.shape, dtype=int)

    adjacent_sudoku[explained_row, :] = given[explained_row, :]
    adjacent_sudoku[:, explained_col] = given[:, explained_col]
    adjacent_sudoku[explained_block_row:explained_block_row+block_offset, explained_block_col:explained_block_col+block_offset] = given[explained_block_row:explained_block_row+block_offset, explained_block_col:explained_block_col+block_offset]

    bv_adjacent_facts_same_value = 0
    bv_adjacent_facts_other_value = [con_map[c] for c in get_sudoku_facts(bvars, adjacent_sudoku)]

    non_adjacent_sudoku = np.array(given)
    non_adjacent_sudoku[explained_row, :] = 0
    non_adjacent_sudoku[:, explained_col] = 0
    non_adjacent_sudoku[explained_block_row:explained_block_row+block_offset, explained_block_col:explained_block_col+block_offset] = 0

    non_adjacent_sudoku_same_value = np.array(non_adjacent_sudoku)
    non_adjacent_sudoku_same_value[given != value_explained] = 0

    non_adjacent_sudoku_other_value = np.array(non_adjacent_sudoku)
    non_adjacent_sudoku_other_value[given == value_explained] = 0

    bv_other_facts_same_value = [con_map[c] for c in get_sudoku_facts(bvars, non_adjacent_sudoku_same_value)]
    bv_other_facts_other_value = [con_map[c] for c in get_sudoku_facts(bvars, non_adjacent_sudoku_other_value)]

    sudoku_adjacent_col = np.zeros(given.shape, dtype=int)
    sudoku_adjacent_col[:, explained_col] = given[:, explained_col]
    sudoku_adjacent_row = np.zeros(given.shape, dtype=int)
    sudoku_adjacent_row[explained_row, :] = given[explained_row, :]
    sudoku_adjacent_block = np.zeros(given.shape, dtype=int)
    sudoku_adjacent_block[explained_block_row:explained_block_row+block_offset, explained_block_col:explained_block_col+block_offset] = given[explained_block_row:explained_block_row+block_offset, explained_block_col:explained_block_col+block_offset]

    bv_adjacent_block_facts = [con_map[c] for c in get_sudoku_facts(bvars, sudoku_adjacent_block)]
    bv_adjacent_row_facts = [con_map[c] for c in get_sudoku_facts(bvars, sudoku_adjacent_row)]
    bv_adjacent_col_facts = [con_map[c] for c in get_sudoku_facts(bvars, sudoku_adjacent_col)]
    bv_adjacent_constraints = [bv_adjacent_row_cons, bv_adjacent_col_cons, bv_adjacent_block_cons]

    bv_other_constr = all_bv_constraints - set(bv_adjacent_constraints)
    bv_other_facts = all_bv_facts - set(bv_adjacent_facts_other_value)

    feature_obj_map = {
        'number_constraints': cp.sum(all_bv_constraints),
        'number_facts': cp.sum(all_bv_facts),
        ## ALL constraints
        'number_block_cons': cp.sum(all_bv_blocks),
        'number_col_cons': cp.sum(all_bv_cols),
        'number_row_cons': cp.sum(all_bv_rows),
        ## ADJACENT COnstraints
        'adjacent_row_used': bv_adjacent_row_cons,
        'adjacent_col_used': bv_adjacent_col_cons,
        'adjacent_block_used': bv_adjacent_block_cons,
        ## OTHER CONSTRAINTS
        'other_row_cons': cp.sum(bv_other_row_cons),
        'other_col_cons': cp.sum(bv_other_col_cons),
        'other_block_cons': cp.sum(bv_other_block_cons),
        ## ADJACENT FACTS
        'number_adjacent_facts_other_value': cp.sum(bv_adjacent_facts_other_value),
        ## OTHER FACTS
        'number_other_facts_same_value': cp.sum(bv_other_facts_same_value),
        'number_other_facts_other_value': cp.sum(bv_other_facts_other_value),
        ## OTHER
        "number_adjacent_block_facts": cp.sum(bv_adjacent_block_facts),
        "number_adjacent_col_facts": cp.sum(bv_adjacent_col_facts),
        "number_adjacent_constraints": cp.sum(bv_adjacent_constraints),
        # "number_adjacent_facts": cp.sum(bv_adjacent_facts_other_value) + bv_adjacent_facts_same_value,
        "number_adjacent_row_facts": cp.sum(bv_adjacent_row_facts),
        "number_other_constraints": cp.sum(bv_other_constr),
        "number_other_facts": cp.sum(bv_other_facts)
    }

    feature_obj_map_norm = {
        'number_constraints': all_bv_constraints,
        'number_facts': all_bv_facts,
        ## ALL constraints
        'number_block_cons': all_bv_blocks,
        'number_col_cons': all_bv_cols,
        'number_row_cons': all_bv_rows,
        ## ADJACENT COnstraints
        'adjacent_row_used': bv_adjacent_row_cons,
        'adjacent_col_used': bv_adjacent_col_cons,
        'adjacent_block_used': bv_adjacent_block_cons,
        ## OTHER CONSTRAINTS
        'other_row_cons': bv_other_row_cons,
        'other_col_cons': bv_other_col_cons,
        'other_block_cons': bv_other_block_cons,
        ## ADJACENT FACTS
        'number_adjacent_facts_other_value': bv_adjacent_facts_other_value,
        ## OTHER FACTS
        'number_other_facts_same_value': bv_other_facts_same_value,
        'number_other_facts_other_value': bv_other_facts_other_value,
        ## OTHER
        "number_adjacent_block_facts": bv_adjacent_block_facts,
        "number_adjacent_col_facts": bv_adjacent_col_facts,
        "number_adjacent_constraints": bv_adjacent_constraints,
        # "number_adjacent_facts": bv_adjacent_facts_other_value + bv_adjacent_facts_same_value,
        "number_adjacent_row_facts": bv_adjacent_row_facts,
        "number_other_constraints": bv_other_constr,
        "number_other_facts": bv_other_facts
    }

    feature_obj_map_norm = {key: value if isinstance(value, (list, set)) else [value] for key, value in feature_obj_map_norm.items()}

    return feature_obj_map,feature_obj_map_norm


def get_integer_facts(bvars, explanation, hint_type=HintType.FACT.name):
    _, nvals = bvars.shape
    assert bvars.shape == (nvals*nvals, nvals), f"Shape of bvars should be {(nvals*nvals, nvals)} but got {bvars.shape}"
    bvars = bvars.reshape((nvals, nvals, nvals))  # 3D cube with boolean vars
    explanation = set(explanation)

    facts = [
        [hint_type, [row, col, val+1]]
        for row in range(nvals)
        for col in range(nvals)
        for val in range(nvals)
        if bvars[row, col, val] in explanation
    ]

    return facts


def featurize_expl_sudoku(expl_cons,features_map,cons_map):
    featurized_expl = {feature:0 for feature in OBJECTIVES_SUDOKU}
    for cons in expl_cons:
        assum = cons_map[cons]
        for feature in OBJECTIVES_SUDOKU:
            if features_map[feature]!=0:
                if isinstance(features_map[feature],_BoolVarImpl):
                    if str(assum)==str(features_map[feature]):
                        featurized_expl[feature] = 1
                else:
                    str_assum = [str(el) for el in features_map[feature].args]
                    if str(assum) in str_assum:
                        featurized_expl[feature] += 1
    return featurized_expl


