import os
import sys
import json
import pandas as pd
import argparse
import numpy as np
import sys


def read_results(result_file):
    with open(result_file, 'r') as f:
        results = json.load(f)
        for result in results:
            result['target_model'] = result['target_config_path'].split('/')[-1].split('.')[0]
            result['tl_dataset'] = result['target_config_path'].split('/')[-3]
        return pd.read_json(json.dumps(results))

def analyze_attack_results(attack_results_root_dir):
    def get_atatck_results(query_results):
        if query_results['tl_dataset'][0] in ['MNIST', 'CIFAR10', 'STL10']:
            c = 10
        elif query_results['tl_dataset'][0] in ['CIFAR100']:
            c = 100
        else:
            c = 2
        tau = np.max([1.5/c, 0.3])

        result = {}

        result['target_model'] = query_results['target_model'][0]
        result['target_model_config_path'] = query_results['target_config_path'][0]

        selected_query_results = query_results[['label_similarity', 'attack_model_name']]
        max_label_simialrity = selected_query_results['label_similarity'].max()
        print("The target model is {}".format(result['target_model']))
        if max_label_simialrity < tau:
            print("The teacher model is unkown")
            result['label_similarity'] = 0
            result['predicted_teacher_model'] = "Nil"
        else:
            result['label_similarity'] = max_label_simialrity
            result['predicted_teacher_model'] = selected_query_results['attack_model_name'][selected_query_results['label_similarity'].idxmax()]
            print("The teacher model is possibly {}".format(result['predicted_teacher_model']))
        return pd.DataFrame([result])


    results = None
    counter = 0
    for root, dirs, files in os.walk(attack_results_root_dir):
        if len(dirs) == 0:
            sub_result_dir = root
            for result_file in os.listdir(sub_result_dir):
                result_path = os.path.join(sub_result_dir, result_file)
                query_results = read_results(result_path)
                new_results = get_atatck_results(query_results)

                if results is None:
                    results = new_results.copy()
                    counter += 1
                else:
                    results = results.append(new_results, ignore_index=True)
                    counter += 1
        """
        else:
            for sub_dir in dirs:
                sub_result_dir = os.path.join(root, sub_dir)
                for result_file in os.listdir(sub_result_dir):
                    print(result_file)
                    print(counter)
                    result_path = os.path.join(sub_result_dir, result_file)
                    if os.path.isdir(result_path):
                        continue
                    new_results = read_results(result_path)
                    if results is None:
                        results = new_results.copy()
                        counter += 1
                    else:
                        results = results.append(new_results, ignore_index=True)
                        counter += 1
        """

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze the model fingerprinting attack.")
    parser.add_argument("--attack_results_root_dir", 
                        type=str,
                        help="Directory storing the attack results.")
    parser.add_argument("--analysis_output", 
                        type=str,
                        help="Output file of the analysis.")
    args = parser.parse_args()
    analysis_results = analyze_attack_results(args.attack_results_root_dir)
    analysis_results.to_csv(args.analysis_output, index=False)
