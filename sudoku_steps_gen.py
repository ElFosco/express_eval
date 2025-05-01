import argparse
import json

from utils.constants import OBJECTIVES_SUDOKU

from utils.utils_sudoku import generate_steps_sudoku, load_sudokus_from_json, generate_top_k_expl

parser = argparse.ArgumentParser(description='Runner sudokus')
parser.add_argument('-u', '--user', type=str, help='user', required=False, default=0)
parser.add_argument('-n', '--normalization', type=int, help='normalization', required=False, default=1)
args = parser.parse_args()
no_users = args.user
normalization = int(args.normalization)

with open("data/weights/sudoku/weights.json", "r") as json_file:
    weights = json.load(json_file)
sudokus = load_sudokus_from_json()

#GENERATION GROUND TRUTH STEPS USER
if no_users=='smus':
    weights_user = None
elif no_users!='real':
    no_users = int(no_users)
    weights_user = dict(zip(OBJECTIVES_SUDOKU, weights[no_users]))
else:
    weights_user = {
        'number_facts': 1,
        'number_constraints': 20,
    }

normalization= {el:1 for el in ['number_constraints','number_facts']}
generate_steps_sudoku(grid=sudokus[30].copy(), feature_weights=weights_user, is_tqdm=True, normalization=normalization,
                      to_return=False,sequential_sudoku=30,no_user=no_users)

# expl_1_featurized = {k: v for k, v in expl_1.items() if k in OBJECTIVES_SUDOKU}
# expl_2, constraints_2 = generate_steps_sudoku(grid=sudokus[30].copy(), feature_weights=weights_user, counter=1,
#                                               diversify=True, obj_values_gt=expl_1_featurized,
#                                               is_tqdm=False, not_allowed=expl_1_featurized,
#                                               normalization=False, fact_to_exp=[0,0])
# print(expl_1)
# print(expl_2)