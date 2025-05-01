import ast
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.lg_problems import LGProblem
from utils.constants import OBJECTIVE_NORMALIZED_SUDOKU, OBJECTIVES_NORMALIZED_LGP
from utils.utils import create_folders
from utils.utils_classes import Oracle
from utils.utils_lgp import generate_steps_lgps
from utils.utils_sudoku import generate_steps_sudoku


def iter_new_regret(run):

    with open("data/weights/sudoku/weights.json", "r") as json_file:
        weights_user = json.load(json_file)

    for subdir_run in sorted(os.listdir(run)):
        print(subdir_run)
        folder_run = os.path.join(run, subdir_run)
        # Step 3: extract user number from the folder name
        match = re.search(r"user_(\d+)", folder_run)
        user_number = int(match.group(1))
        match = re.search(r'results_RQ1/([^/_]+)', run)
        problem_type = match.group(1)

        second_folder = run.split('/')[2]
        # Extract the number after the last underscore
        match = re.search(r'_(\d+)$', second_folder)
        norm = int(match.group(1)) if match else 2

        if problem_type == 'lgps':
            problem_type = 'lgp'
        oracle = Oracle(weights=weights_user[user_number], problem=problem_type)

        new_dir = folder_run.replace("results_RQ1", "results_new_RQ1", 1)
        if problem_type=='sudoku':
            df_steps_evaluation = pd.read_csv(f'data/gt_{problem_type}/normalization_{0}/sudoku_user_{user_number}_sudoku_30.csv',index_col=False)
        else:
            problem_type = 'lgp'
            df_steps_evaluation = pd.read_csv(f'data/gt_{problem_type}/normalization_{0}/lgp_user_{user_number}_problem_1.csv',index_col=False)
        re_compute_regret(folder_run,problem_type,oracle,new_dir,df_steps_evaluation,norm)


def re_compute_regret(folder,problem_type,oracle,output_location,df_steps_evaluation,norm):
    create_folders(output_location)
    if problem_type=='sudoku':
        keys = [
            'number_adjacent_facts_other_value','number_other_facts_same_value','number_other_facts_other_value',
            'adjacent_col_used','adjacent_row_used','adjacent_block_used','other_col_cons','other_row_cons','other_block_cons',
            'number_adjacent_row_facts','number_adjacent_col_facts','number_adjacent_block_facts'
        ]

        df_for_weights = pd.read_csv(f'{folder}/regret.csv')
        weights = ast.literal_eval(df_for_weights.iloc[1, 2])
        weights_learned = dict(zip(keys, weights))
        df_for_norm = pd.read_csv(f'{folder}/dataset.csv')
        if norm ==0:
            norm = {k: 1 for k in keys}
        if norm ==1:
            norm = {k: OBJECTIVE_NORMALIZED_SUDOKU[k] for k in keys}
        if norm==2:
            features_1_list = eval(df_for_norm.loc[99, 'features 1'])
            features_2_list = eval(df_for_norm.loc[99, 'features 2'])
            max_list = [max(f1, f2, 1) for f1, f2 in zip(features_1_list, features_2_list)]
            norm = {k: el for k,el in zip(keys,max_list)}
        if norm ==3:
            features_1_lists = df_for_norm['features 1'].apply(ast.literal_eval)[:100]
            features_2_lists = df_for_norm['features 2'].apply(ast.literal_eval)[:100]
            combined = pd.concat([features_1_lists, features_2_lists], ignore_index=True)
            max_list = [max(1, *values) for values in zip(*combined)]
            norm = {k: el for k, el in zip(keys, max_list)}
        evaluate(problem_type,df_steps_evaluation,weights_learned,norm,oracle,output_location)

    if problem_type == 'lgp':
        keys = [
            'adjacent_negative_facts','adjacent_facts_from_clue','adjacent_facts_from_bijectivity',
            'adjacent_facts_from_transitivity','other_negative_facts','other_positive_facts','adjacent_clue',
            'adjacent_bijectivity','adjacent_transitivity','other_clue','other_transitivity','other_bijectivity'
        ]

        df_for_weights = pd.read_csv(f'{folder}/regret.csv')
        weights = ast.literal_eval(df_for_weights.iloc[1, 2])
        weights_learned = dict(zip(keys, weights))
        df_for_norm = pd.read_csv(f'{folder}/dataset.csv')
        if norm == 0:
            norm = {k: 1 for k in keys}
        if norm ==1:
            norm = {k: OBJECTIVES_NORMALIZED_LGP[k] for k in keys}
        if norm==2:
            features_1_list = eval(df_for_norm.loc[99, 'features 1'])
            features_2_list = eval(df_for_norm.loc[99, 'features 2'])
            max_list = [max(f1, f2, 1) for f1, f2 in zip(features_1_list, features_2_list)]
            norm = {k: el for k, el in zip(keys, max_list)}         #must be changed
        if norm==3:
            features_1_lists = df_for_norm['features 1'].apply(ast.literal_eval)[:100]
            features_2_lists = df_for_norm['features 2'].apply(ast.literal_eval)[:100]
            combined = pd.concat([features_1_lists, features_2_lists], ignore_index=True)
            max_list = [max(1, *values) for values in zip(*combined)]
            norm = {k: el for k, el in zip(keys, max_list)}
        instance_evaluation = []
        maker = LGProblem(type=1)
        [], constraints, facts, explainable_facts, dict_constraint_type, dict_constraints_involved, rels_visualization, dict_constraints_clues = maker.make_model()
        instance_evaluation.append([facts,constraints,explainable_facts,dict_constraint_type,dict_constraints_involved])
        evaluate(problem_type, df_steps_evaluation, weights_learned, norm, oracle, output_location, instance_evaluation)





def evaluate(problem_type,df_steps_evaluation,weights_learned,normalization_values,oracle,output_location,
             instance_evaluation=None):
    print('Start evaluation:')
    predicted = []
    if problem_type == 'sudoku':
        for index, row in tqdm(df_steps_evaluation.iterrows(), total=len(df_steps_evaluation)):
            sudoku_eval = row['grid'].replace(' ', ', ')
            sudoku_eval = np.array(eval(sudoku_eval))
            parsed_list = eval(row['hint_derived'])
            exp,_,_ = generate_steps_sudoku(to_return=True, grid=sudoku_eval,
                                            feature_weights=weights_learned.copy(),no_user=1,
                                            sequential_sudoku=-1,normalization=normalization_values,
                                            is_tqdm=False)
                                            #fact_to_exp=[explained_row,explained_col])
            predicted.append(exp)

        df_predicted_steps = pd.DataFrame(predicted)
        dict_features_weights = oracle.get_dict_feature_weights()

        weighted_df_predicted = df_predicted_steps[dict_features_weights.keys()].mul(dict_features_weights)
        weighted_df_predicted = weighted_df_predicted.sum(axis=1)

        weighted_df_gt = df_steps_evaluation[dict_features_weights.keys()].mul(dict_features_weights)
        weighted_df_gt = weighted_df_gt.sum(axis=1)
        difference_df_abs = (weighted_df_predicted - weighted_df_gt)
        difference_df = (weighted_df_predicted - weighted_df_gt) / weighted_df_gt


        try:
            assert (difference_df >= 0).all().all(), "Some values in difference_df are not positive"
        except AssertionError as e:
            create_folders('./errors')
            error_filename = f"./errors/error_log.txt"
            problematic_rows = difference_df < 1e-2  # Boolean mask for problematic rows
            with open(error_filename, "w") as f:
                f.write(str(e))
                f.write("\nDifference DF (only problematic rows):\n")
                f.write(difference_df[problematic_rows].to_string())

                f.write("\n\nPredicted Steps DF (only problematic rows):\n")
                f.write(df_predicted_steps.loc[problematic_rows, dict_features_weights.keys()].to_string())

                f.write("\n\nGround Truth Steps DF (only problematic rows):\n")


        difference_df = ((weighted_df_predicted - weighted_df_gt) / weighted_df_gt).clip(lower=0)

        # Compute the average of the differences
        regret = max(0,difference_df.mean().mean())
        weights = str([float(value) for value in weights_learned.values()])
        regret_list = [[weights,regret]]
        df = pd.DataFrame(regret_list, columns=['weights', 'regret'])
        df.to_csv(f'{output_location}' + f'/regret.csv')
        print(f'Regret: {regret}')
    if problem_type == 'lgp':
        facts = instance_evaluation[0][0].copy()
        constraints = instance_evaluation[0][1].copy()
        explainable_facts = instance_evaluation[0][2].copy()
        dict_constraint_type = instance_evaluation[0][3].copy()
        dict_adjacency = instance_evaluation[0][4].copy()

        for index, row in tqdm(df_steps_evaluation.iterrows(), total=len(df_steps_evaluation)):
            parsed_literal = row['explained'].lstrip("~")
            to_explain = [next((el for el in explainable_facts if el.name == parsed_literal), None)]

            exp,_,_ = generate_steps_lgps(facts.copy(), constraints, explainable_facts.copy(),
                                          dict_constraint_type, dict_adjacency,
                                          feature_weights= weights_learned, is_tqdm=False,
                                          normalization = normalization_values, to_return=True,
                                          counter=1)
            predicted.append(exp)
            if row['explained'].startswith("~"):
                facts.append(~to_explain[0])
            else:
                facts.append(to_explain[0])
            explainable_facts.remove(to_explain[0])
        df_predicted_steps = pd.DataFrame(predicted)
        dict_features_weights = oracle.get_dict_feature_weights()

        weighted_df_predicted = df_predicted_steps[dict_features_weights.keys()].mul(dict_features_weights)
        weighted_df_predicted = weighted_df_predicted.sum(axis=1)

        weighted_df_gt = df_steps_evaluation[dict_features_weights.keys()].mul(dict_features_weights)
        weighted_df_gt =  weighted_df_gt.sum(axis=1)
        difference_df = (weighted_df_predicted - weighted_df_gt)/weighted_df_gt


        # Compute the average of the differences
        regret = max(0,difference_df.mean().mean())
        weights = str([float(value) for value in weights_learned.values()])
        regret_list = [[weights, regret]]
        df = pd.DataFrame(regret_list, columns=['weights', 'regret'])
        df.to_csv(f'{output_location}' + f'/regret.csv')
        print(f'Regret: {regret}')
