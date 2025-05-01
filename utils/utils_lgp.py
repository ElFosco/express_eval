from collections import defaultdict
from time import time
import copy
from cpmpy.transformations.normalize import toplevel_list
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView

from utils.constants import WEIGHTED_DIVERSIFY, CPUCB_DIVERSIFY, OBJECTIVES_LGP
from utils.utils import split_ocus_assum, leftover, create_folders, ucb_dictionary, compute_normalization
from utils.utils_classes import HintType


def generate_steps_lgps(facts, constraints, to_explain, dict_constraint_type, dict_adjacency, sequential_lgp=-1,
                        single_fact=None,feature_weights=None,counter=None, diversify=False, gamma=0, obj_values_gt=None,
                        no_user=None, is_tqdm=True, normalization = [], to_return=False,
                        trade_offs=None,iterations=None):

    if feature_weights==None:
        feature_weights = {
            'number_facts': 1,
            'number_constraints': 1,
        }
    info = []
    if counter == None:
        counter = len(to_explain)
    if is_tqdm:
        to_consider = tqdm(range(counter))
    else:
        to_consider = range(counter)
    for _ in to_consider:
        generator = get_split_ocus_hint_lgp(facts, constraints, to_explain, feature_weights, dict_constraint_type,
                                            dict_adjacency,single_fact=single_fact ,time_limit=3600, hs_solver_name="gurobi",
                                            diversify=diversify, obj_values_gt=obj_values_gt,gamma=gamma, normalization=normalization,
                                            solvername='exact',trade_offs=trade_offs,iterations=iterations)
        time_start = time()
        exp, constraints_expl,features = next(generator)
        time_end = time()

        if isinstance(exp['explained'], NegBoolView):
            to_explain.remove(~exp['explained'])
        else:
            to_explain.remove(exp['explained'])
        exp['remaining'] = to_explain.copy()

        if sequential_lgp >= 0:
            facts.append(exp['explained'])
            exp['time'] = time_end - time_start
            info.append(exp)
            df = pd.DataFrame(info)
            if not to_return:
                create_folders(f"./data/gt_lgp/normalization_{normalization}")
                df.to_csv(f"./data/gt_lgp/normalization_{normalization}/lgp_user_{no_user}_problem_{sequential_lgp}.csv", index=False)
        else:
            return exp, constraints_expl, features
    return df

def new_generate_steps_lgps(facts, constraints, to_explain, dict_constraint_type, dict_adjacency, sequential_lgp=-1,
                        single_fact=None,feature_weights=None,counter=None, diversify=False, gamma=0, obj_values_gt=None,
                        no_user=None, is_tqdm=True, normalization = [],
                        trade_offs=None,iterations=None):

    if feature_weights==None:
        feature_weights = {
            'number_facts': 1,
            'number_constraints': 1,
        }
    info = []
    if counter == None:
        counter = len(to_explain)
    if is_tqdm:
        to_consider = tqdm(range(counter))
    else:
        to_consider = range(counter)
    for _ in to_consider:
        generator = get_split_ocus_hint_lgp(facts, constraints, to_explain, feature_weights, dict_constraint_type,
                                            dict_adjacency,single_fact=single_fact ,time_limit=3600, hs_solver_name="gurobi",
                                            diversify=diversify, obj_values_gt=obj_values_gt,gamma=gamma, normalization=normalization,
                                            solvername='exact',trade_offs=trade_offs,iterations=iterations)
        time_start = time()
        exp_1, constraints_expl,features = next(generator)
        time_end = time()

        if isinstance(exp_1['explained'], NegBoolView):
            to_explain.remove(~exp_1['explained'])
        else:
            to_explain.remove(exp_1['explained'])
        exp_1['remaining'] = to_explain.copy()


        exp_1['time'] = time_end - time_start
        info.append(exp_1)
        df = pd.DataFrame(info)
        create_folders(f"./data/gt_lgp/normalization_{normalization}")
        df.to_csv(f"./data/gt_lgp/normalization_{normalization}/lgp_user_{no_user}_problem_{sequential_lgp}.csv", index=False)

        generator = get_split_ocus_hint_lgp(facts, constraints, to_explain, feature_weights, dict_constraint_type,
                                            dict_adjacency, single_fact=single_fact, time_limit=3600,
                                            hs_solver_name="gurobi",
                                            diversify=diversify, obj_values_gt=obj_values_gt, gamma=gamma,
                                            normalization=normalization,
                                            solvername='exact', trade_offs=trade_offs, iterations=iterations)
        time_start = time()
        exp_2, constraints_expl, features = next(generator)
        time_end = time()


        exp_2['remaining'] = to_explain.copy()
        exp_2['time'] = time_end - time_start
        info.append(exp_2)
        df = pd.DataFrame(info)
        create_folders(f"./data/gt_lgp/normalization_{normalization}")
        df.to_csv(f"./data/gt_lgp/normalization_{normalization}/lgp_user_{no_user}_problem_{sequential_lgp}.csv",
                  index=False)

        facts.append(exp_1['explained'])
    return df





def get_split_ocus_hint_lgp(facts, constraints, to_explain, feature_weights, dict_constraint_type,
                            dict_adjacency,single_fact=None, gamma=0, hs_solver_name='gurobi', solvername='exact', param_dict=dict(),
                            time_limit=3600, diversify=False,obj_values_gt=None,normalization = [],
                            trade_offs=None,iterations=None):
    # definition weights_exploration
    weights_exploration = {}
    if diversify in WEIGHTED_DIVERSIFY or diversify == 'w_hamming':
        for obj in feature_weights:
            weights_exploration[obj] = feature_weights[obj]


    if diversify == 'ucb':
        ranked_trade_offs = ucb_dictionary(trade_offs.copy(), feature_weights, iterations)
        for obj in feature_weights:
            weights_exploration[obj] = 0
            feature_weights[obj] = ranked_trade_offs[obj]


    elif diversify in CPUCB_DIVERSIFY or diversify == 'cpucb_hamming':
        # max_v = max(feature_weights.values())
        # feature_weights = {k: v / max_v for k, v in feature_weights.items()}
        ranked_trade_offs = ucb_dictionary(trade_offs, feature_weights, iterations)
        for obj in feature_weights:
            weights_exploration[obj] = ranked_trade_offs[obj]



    model = cp.Model()
    model += facts + constraints
    assert cp.Model(constraints + facts).solveAll(solution_limit=2) == 1, "Found more than 1 solution, or model is UNSAT!!"

    dict_features = keep_explained_el(dict_adjacency,facts,len(feature_weights.keys())==1,constraints,to_explain)

    map_explained_features_1 = {}
    objectives = []
    hs_hard_explained = {}
    if normalization is None:
        objective_normalized_lgps = {}
        objective_normalized_lgps['all'] = 1
    else:
        objective_normalized_lgps = normalization
    if single_fact is not None:
        # to_explain_value_for_norm = get_facts_to_explain(to_explain)
        # soft_norm = facts + constraints + to_explain_value_for_norm
        # assump_norm = cp.boolvar(shape=len(soft_norm), name="assumption")
        # con_map_norm = dict(zip(soft_norm, assump_norm))
        #
        # for bv_explained in to_explain_value_for_norm:
        #     tmp_for_norm[con_map_norm[bv_explained]] = dict_features[bv_explained]  # here we need the d.v.

        # objective_normalized_lgps = compute_normalization(tmp_for_norm, normalization,problem_type='lgp')

        to_explain_value = get_facts_to_explain(single_fact)
        soft = facts + constraints + to_explain_value
        assump = cp.boolvar(shape=len(soft), name="assumption")
        assum_map = dict(zip(assump, soft))
        oneof_idxes = np.arange(len(facts + constraints), len(facts + constraints + to_explain_value))
        con_map = dict(zip(soft, assump))
        for bv_explained in to_explain_value:
            map_explained_features_1[con_map[bv_explained]] = dict_features[bv_explained]

    else:
        to_explain_value = get_facts_to_explain(to_explain)
        soft = facts + constraints + to_explain_value
        assump = cp.boolvar(shape=len(soft), name="assumption")
        con_map = dict(zip(soft, assump))
        assum_map = dict(zip(assump, soft))
        oneof_idxes = np.arange(len(facts + constraints), len(facts + constraints + to_explain_value))
        for bv_explained in to_explain_value:
            map_explained_features_1[con_map[bv_explained]] = dict_features[bv_explained]


    ## Reify assumption variables
    hard = [assump.implies(soft)]  # each assumption variable implies a candidate

    if not diversify:
        obj_values_gt = None

    for bv_explained in to_explain_value:

        objective = 0
        tmp = 0
        all_feature_weights = []
        all_used_features = []
        normalized_weights = {}
        integer_normalization = {}


        for feature, feature_weight in feature_weights.items():
            map_explained_features_1[con_map[bv_explained]][feature] = sum(con_map[el] for el in map_explained_features_1[con_map[bv_explained]][feature])

        for feature, feature_weight in feature_weights.items():
            all_feature_weights.append(feature_weight)
            all_used_features.append(map_explained_features_1[con_map[bv_explained]][feature])
            if diversify == 'baseline' or diversify == 'disjunction' or diversify == 'coverage' or diversify == 'coverage_sum':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_lgps[feature])
                integer_normalization[feature] = int(1e5 / objective_normalized_lgps[feature])
                tmp += map_explained_features_1[con_map[bv_explained]][feature] * normalized_weights[feature]
            elif diversify in WEIGHTED_DIVERSIFY or diversify in CPUCB_DIVERSIFY or diversify == 'ucb':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_lgps[feature])
                integer_normalization[feature] = int(1e5 * weights_exploration[feature] / objective_normalized_lgps[feature])
                tmp += map_explained_features_1[con_map[bv_explained]][feature] * normalized_weights[feature]
            elif diversify == 'hamming':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_lgps[feature])
                integer_normalization[feature] = int(1e5)
                tmp += map_explained_features_1[con_map[bv_explained]][feature] * normalized_weights[feature]
            elif diversify == 'w_hamming' or diversify == 'lex_hamming' or diversify == 'cpucb_hamming':
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_lgps[feature])
                integer_normalization[feature] = int(1e5 * weights_exploration[feature])
                tmp += map_explained_features_1[con_map[bv_explained]][feature] * normalized_weights[feature]
            else: #no diversification
                normalized_weights[feature] = int(1e5 * feature_weight / objective_normalized_lgps[feature])
                tmp += map_explained_features_1[con_map[bv_explained]][feature] * normalized_weights[feature]
        if gamma is None:
            objective += tmp
            gamma = 1
        else:
            objective += tmp * (1 - gamma)
        if diversify == 'baseline':
            different = sum(cp.max(integer_normalization[feature]*(map_explained_features_1[con_map[bv_explained]][feature] - value),
                                   integer_normalization[feature]*(value - map_explained_features_1[con_map[bv_explained]][feature])) for feature, value in obj_values_gt.items())
            objective += -gamma * (different)
            temp_list = [map_explained_features_1[con_map[bv_explained]][feature] != obj_values_gt[feature] for feature in obj_values_gt]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'ucb':
            temp_list = [
                (map_explained_features_1[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features_1[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'disjunction' or diversify == 'w_disjunction' or diversify == 'cpucb_disjunction' or diversify == 'lex_disjunction':
            different = sum(cp.max(integer_normalization[feature] * (map_explained_features_1[con_map[bv_explained]][feature] - value),
                                   integer_normalization[feature] * (value - map_explained_features_1[con_map[bv_explained]][feature])) for
                            feature, value in obj_values_gt.items())
            objective += -gamma * (different)
            temp_list = [
                (map_explained_features_1[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features_1[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'coverage' or diversify == 'w_coverage' or diversify == 'cpucb_coverage' or diversify == 'lex_coverage':
            different = cp.max(integer_normalization[feature] * (value - map_explained_features_1[con_map[bv_explained]][feature]) for feature, value in
                               obj_values_gt.items())
            objective += - gamma * different
            temp_list = [
                (map_explained_features_1[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features_1[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'coverage_sum' or diversify == 'w_coverage_sum' or diversify == 'cpucb_coverage_sum' or diversify == 'lex_coverage_sum':
            different = sum(integer_normalization[feature] * (value - map_explained_features_1[con_map[bv_explained]][feature]) for feature, value in
                            obj_values_gt.items())
            objective += - gamma * different
            temp_list = [
                (map_explained_features_1[con_map[bv_explained]][feature] < obj_values_gt[feature])
                if feature_weights[feature] >= 0
                else (map_explained_features_1[con_map[bv_explained]][feature] > obj_values_gt[feature])
                for feature in obj_values_gt
            ]
            hs_hard_explained[con_map[bv_explained]] = cp.any(temp_list)
        if diversify == 'hamming' or diversify == 'w_hamming' or diversify == 'cpucb_hamming' or diversify == 'lex_hamming':
            different = sum(
                integer_normalization[feature] * (cp.sum([map_explained_features_1[con_map[bv_explained]][feature]]) != value) for feature, value in
                obj_values_gt.items())
            objective += - gamma * different
            # temp_list = [
            #     (map_explained_features_1[con_map[bv_explained]][feature] < obj_values_gt[feature])
            #     if feature_weights[feature] >= 0
            #     else (map_explained_features_1[con_map[bv_explained]][feature] > obj_values_gt[feature])
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
        list_strings = [str(el) for el in to_explain_value]
        expl_cons_str = [str(el) for el in expl_cons]
        if expl_cons is not None:
            for el in expl_cons:
                if str(el) in list_strings:
                    explained = ~el
                    expl_cons.pop(expl_cons_str.index(str(el)))
                    break
            features_map = map_explained_features_1[con_map[~explained]]
            featurized_expl = featurize_expl_lgp(expl_cons, features_map, con_map)

            expl = {
                'explained': explained,
                "hint_constraints": expl_cons,
            }
            expl.update(featurized_expl)

            yield expl,expl_cons,featurized_expl

def featurize_expl_lgp(expl_cons,features_map,cons_map):
    featurized_expl = {feature:0 for feature in features_map.keys()}
    for cons in expl_cons:
        assum = cons_map[cons]
        for feature in features_map:
            if features_map[feature]!=0:
                if isinstance(features_map[feature],_BoolVarImpl):
                    if str(assum)==str(features_map[feature]):
                        featurized_expl[feature] = 1
                else:
                    str_assum = [str(el) for el in features_map[feature].args]
                    if str(assum) in str_assum:
                        featurized_expl[feature] += 1
    return featurized_expl




def get_facts_to_explain(explainable_facts):
    to_explain = []
    for ex in explainable_facts:
        if ex.value():
            to_explain.append(~ex)
        else:
            to_explain.append(ex)
    return to_explain

# def map_lgp_features(facts,explained,dict_constraint_type,dict_adjacency):


# def map_lgp_features(facts,explained,dict_constraint_type,dict_adjacency):
#     dict_features_constraints = defaultdict(list)
#     # Initialize all keys with an empty list
#     for key in OBJECTIVES_LGP:
#         dict_features_constraints[key] = []
#
#
#     if isinstance(explained, NegBoolView):
#         explained = ~explained
#     top_level_facts = toplevel_list(facts, merge_and=False)
#     negative_facts = []
#     positive_facts = []
#
#
#     for f in top_level_facts:
#         if isinstance(f, NegBoolView):
#             negative_facts.append(~f)
#         else:
#             positive_facts.append(f)
#     for nf in negative_facts:
#         items = is_fact_adjacent(explained, nf, dict_constraint_type, dict_adjacency)
#         if len(items) > 0 :
#             dict_features_constraints['adjacent_negative_facts'].append(~nf)
#             for type in items:
#                 dict_features_constraints[f'adjacent_facts_from_{type}'].append(~nf)
#         else:
#             dict_features_constraints['other_negative_facts'].append(~nf)
#     for pf in positive_facts:
#         items = is_fact_adjacent(explained, pf, dict_constraint_type, dict_adjacency)
#         if len(items) > 0 :
#             for type in items:
#                 dict_features_constraints[f'adjacent_facts_from_{type}'].append(pf)
#         else:
#             dict_features_constraints['other_positive_facts'].append(pf)
#     for cnstr,type in dict_constraint_type.items():
#         if is_constraint_adjacent(explained,cnstr,dict_adjacency):
#             dict_features_constraints[f'adjacent_{type}'].append(cnstr)
#         else:
#             dict_features_constraints[f'other_{type}'].append(cnstr)
#     return dict_features_constraints
#
#
# def is_fact_adjacent(explained,fact,dict_constraint_type,dict_adjaceny):
#     set_adjacent = set()
#     for constraint in dict_adjaceny:
#         str_list = [str(el) for el in set(dict_adjaceny[constraint])]
#         if str(explained) in str_list and str(fact) in str_list:
#             set_adjacent.add(dict_constraint_type[constraint])
#     return set_adjacent
#
# def is_constraint_adjacent(explained,cnstr,dict_adjaceny):
#     if str(explained) in [str(el) for el in set(dict_adjaceny[cnstr])]:
#         return True
#     else:
#         return False


def generate_images_explanation_lgps(rels_visualization,dict_constraints_clues,facts,file_input,name_folder='./visualization'):
    df = pd.read_csv(file_input, index_col=-1)
    create_folders(name_folder)
    k=0
    previous_facts = facts
    for index, row in df.iterrows():
        mus = row['hint_constraints']
        explained_facts = row['explained']
        visualize_lgp(rels_visualization, dict_constraints_clues, mus, explained_facts,previous_facts,
                      name_folder=name_folder, name_file=f'step_{k}.png')
        previous_facts.append(explained_facts)
        k += 1


def visualize_lgp(rels_visualization,dict_constraints_clues,mus,explained_facts,previous_facts,
                  name_folder='./visualization',name_file='test.png'):
    cmap = plt.get_cmap('Pastel1')
    df = prepare_data_visualization(rels_visualization)
    fig = plt.figure(figsize=(12, 12))  # Maintain the larger figure size
    ax1 = fig.add_subplot(211)
    ax1.axis('tight')
    ax1.axis('off')
    table = ax1.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

    remaining_tables = 0
    steps_y = int(df.shape[0] / 5)
    for i in range(steps_y):
        steps_x = int(df.shape[1] / 5)
        for j in range(steps_x - remaining_tables):
            block_color = cmap(i+(j*3))
            for k in range(1,6):
                for l in range(5):
                    cell = table[(i*5 + k, j*5 + l)]
                    cell.set_facecolor(block_color)
        remaining_tables += 1
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    for fact in previous_facts:
        if fact.startswith('~'):
            str_fact = fact[1:]
        else:
            str_fact = fact
        indices = [(i + 1, j) for i in range(df.shape[0]) for j in range(df.shape[1]) if str_fact == str(df.iat[i, j])]
        if not fact.startswith('~'):
            table[(indices[0])].set_facecolor("green")
        else:
            table[(indices[0])].set_facecolor("red")

    if explained_facts.startswith('~'):
        str_fact = explained_facts[1:]
    else:
        str_fact = explained_facts
    indices = [(i+1, j) for i in range(df.shape[0]) for j in range(df.shape[1]) if str_fact == str(df.iat[i, j])]
    if not explained_facts.startswith('~'):
        table[(indices[0])].set_facecolor("green")
    else:
        table[(indices[0])].set_facecolor("red")
    table[(indices[0])].set_edgecolor("purple")
    table[(indices[0])].set_linewidth(6)


    table.scale(2, 1)
    table.auto_set_font_size(False)

    ax2 = fig.add_subplot(212)
    ax2.axis('tight')
    ax2.axis('off')

    data_for_second_table = [clue for clue in set(dict_constraints_clues.values())]
    df2 = pd.DataFrame(data_for_second_table, columns=['Description'])

    table2 = ax2.table(cellText=df2.values, colLabels=df2.columns, cellLoc='center', loc='center')
    mus = split_expressions(mus)
    for constraint in mus:
        if constraint in dict_constraints_clues:
            clues_associated = dict_constraints_clues[constraint]
            for (i, j), cell in table2.get_celld().items():
                txt = cell.get_text().get_text()
                if txt == clues_associated:
                    cell = table2[(i, j)]
                    cell.set_facecolor('Green')
        else:
            constraint_cleaned = str(constraint).lstrip("~")
            indices = [(i + 1, j) for i in range(df.shape[0]) for j in range(df.shape[1]) if
                       constraint_cleaned == str(df.iat[i, j])]
            table[(indices[0])].set_edgecolor("blue")
            table[(indices[0])].set_linewidth(2)

    # fig.text(0.5, 0.5, mus, ha='center', va='center', fontsize=12,bbox=dict(facecolor='lightgrey',
    #                                                                         edgecolor='black',
    #                                                                         boxstyle='round,pad=0.5'))

    table2.scale(1.5, 1.5)  # Adjust scale as needed
    # Display the table
    #plt.show()
    plt.savefig(f'./{name_folder}/{name_file}', bbox_inches='tight')




def prepare_data_visualization(dataframe_grid):
    dfs = dataframe_grid[0]
    max_size = len(dfs)
    result = pd.concat(dfs, axis=1)
    for df_list in dataframe_grid[1:]:
        iter = len(df_list)
        tmp = pd.concat(df_list, axis=1)
        row_indices = df_list[0].index
        for i in range(iter,max_size):
            column_indices = dataframe_grid[0][i].columns
            df_nd = pd.DataFrame('', index=row_indices, columns=column_indices)
            tmp = pd.concat([tmp,df_nd],axis=1)
        result = pd.concat([result, tmp], axis=0)
    return result



def split_expressions(input_string):
    # Remove the outer brackets
    trimmed = input_string.strip("[]")

    # Initialize variables for tracking
    expressions = []
    current_expr = []
    bracket_level = 0

    # Iterate through each character to split based on nesting levels
    for char in trimmed:
        if char == "[":
            bracket_level += 1
        elif char == "]":
            bracket_level -= 1
        elif char == "," and bracket_level == 0:
            # Join current expression and add to list, then reset
            expressions.append("".join(current_expr).strip())
            current_expr = []
            continue

        # Add character to current expression
        current_expr.append(char)

    # Append the last expression
    expressions.append("".join(current_expr).strip())

    return expressions


def define_dict_adjacency(constraints, facts, explainable_facts, dict_constraint_type, dict_constraint_involved):
    model = cp.Model()
    model += constraints
    model += facts
    model.solve()

    fact_dv = []
    for f in facts:
        if isinstance(f, NegBoolView):
            fact_dv.append(f._bv)
        else:
            fact_dv.append(f)

    all_facts = fact_dv + list(explainable_facts)

    dict_fact = {}

    for fact in explainable_facts:
        key_el = get_facts_to_explain([fact])
        dict_fact_constraint_objective = {}
        dict_fact_fact_objective = {}
        for obj in ['other_bijectivity','other_transitivity','other_clue',
                    'adjacent_transitivity','adjacent_bijectivity','adjacent_clue']:
            dict_fact_constraint_objective[obj] = set()
        for obj in ['adjacent_negative_facts','adjacent_facts_from_clue','adjacent_facts_from_bijectivity',
                    'adjacent_facts_from_transitivity','other_negative_facts','other_positive_facts']:
            dict_fact_fact_objective[obj] = set()

        list_adj_fact = set()
        for constraint in dict_constraint_involved:
            if fact.name in [el.name for el in dict_constraint_involved[constraint]]:
                for fact_adj in dict_constraint_involved[constraint]:
                    if fact.name != fact_adj.name and fact_adj.value():
                        dict_fact_fact_objective[f'adjacent_facts_from_{dict_constraint_type[constraint]}'].add(fact_adj)
                        list_adj_fact.add(fact_adj)
                    elif fact.name != fact_adj.name and not fact_adj.value():
                        dict_fact_fact_objective[f'adjacent_facts_from_{dict_constraint_type[constraint]}'].add(~fact_adj)
                        dict_fact_fact_objective[f'adjacent_negative_facts'].add(~fact_adj)
                        list_adj_fact.add(fact_adj)
                dict_fact_constraint_objective[f'adjacent_{dict_constraint_type[constraint]}'].add(constraint)
            else:
                dict_fact_constraint_objective[f'other_{dict_constraint_type[constraint]}'].add(constraint)

        for f in all_facts:
            if f.name not in [el.name for el in list_adj_fact] and f.value():
                dict_fact_fact_objective['other_positive_facts'].add(f)
            elif f.name not in [el.name for el in list_adj_fact] and not f.value():
                dict_fact_fact_objective['other_negative_facts'].add(~f)

        for type_const in dict_fact_fact_objective:
            if fact.name in [el.name for el in dict_fact_fact_objective[type_const]]:
                dict_fact_fact_objective[type_const].remove(fact)

        dict_fact[key_el[0]] = {**dict_fact_constraint_objective, **dict_fact_fact_objective}

    return dict_fact



def keep_explained_el(dict_fact_fact,facts,smus=False,constraints=None,to_explain=None):
    new_dict_fact_fact = {}
    if not smus:
        new_dict_fact_fact = copy.deepcopy(dict_fact_fact)
        for fact in dict_fact_fact:
            dict_obj_facts = dict_fact_fact[fact]
            for obj in ['adjacent_negative_facts','adjacent_facts_from_clue','adjacent_facts_from_bijectivity',
                        'adjacent_facts_from_transitivity','other_negative_facts','other_positive_facts']:
                names_to_keep = {el.name for el in facts}
                new_dict_fact_fact[fact][obj] = {el for el in dict_obj_facts[obj] if el.name in names_to_keep}
    else:
        facts_to_explain = get_facts_to_explain(to_explain)
        f = facts.copy()
        c = constraints.copy()
        f.extend(c)
        for fact in facts_to_explain:
            new_dict_fact_fact[fact]={}
            new_dict_fact_fact[fact]['all'] = f
    return new_dict_fact_fact





