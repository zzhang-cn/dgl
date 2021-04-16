import pandas as pd
import json
from pathlib import Path
from itertools import product


def get_branch_name_from_hash(hash):
    import subprocess
    process = subprocess.Popen(['git', 'name-rev', '--name-only', hash],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if len(stderr) > 0:
        return hash[:10]
    else:
        return stdout.decode("utf-8") .strip("\n")


def main():
    results_path = Path("../results")
    results_path.is_dir()
    benchmark_json_path = results_path / "benchmarks.json"
    with benchmark_json_path.open() as f:
        benchmark_json = json.load(f)
    machines = [f for f in results_path.glob("*") if f.is_dir()]
    output_results_dict = {}
    for machine in machines:
        # commit_results_dict = {}
        per_machine_result = {}
        commit_results_json_paths = [
            f for f in machine.glob("*") if f.name != "machine.json"]
        for commit in commit_results_json_paths:
            with commit.open() as f:
                commit_result = json.load(f)
            commit_hash = commit_result['commit_hash']
            per_commit_result = {}
            for test_name, result in commit_result['results'].items():
                per_commit_result[test_name] = []
                if result['result'] is None:
                    for test_args in product(*result['params']):
                        per_commit_result[test_name].append(
                            {"params": ", ".join(test_args), "result": None})
                else:
                    for test_args, performance_number in zip(product(*result['params']), result['result']):
                        per_commit_result[test_name].append(
                            {"params": ", ".join(test_args), "result": performance_number})
            per_machine_result[commit_hash] = per_commit_result
        output_results_dict[machine.name] = per_machine_result
    return output_results_dict


def dict_to_csv(output_results_dict):
    with open("../results/benchmarks.json") as f:
        benchmark_conf = json.load(f)
    unit_dict = {}
    for k, v in benchmark_conf.items():
        if k != 'version':
            unit_dict[k] = v['unit']
    result_list = []
    for machine, per_machine_result in output_results_dict.items():
        for commit, test_cases in per_machine_result.items():
            branch_name = get_branch_name_from_hash(commit)
            result_column_name = "number_{}".format(branch_name)
            # per_commit_result_list = []
            for test_case_name, results in test_cases.items():
                for result in results:
                    result_list.append(
                        {"test_name": test_case_name, 'params': result['params'], 'unit': unit_dict[test_case_name], "number": result['result'], 'commit': branch_name, 'machine': machine})
    df = pd.DataFrame(result_list)
    return df


def side_by_side_view(df):
    commits = df['commit'].unique().tolist()
    full_df = df.loc[df['commit'] == commits[0]]
    for commit in commits[1:]:
        per_commit_df = df.loc[df['commit'] == commit]
        full_df: pd.DataFrame = full_df.merge(
            per_commit_df, on=['test_name', 'params', 'machine', 'unit'], how='outer', suffixes=("_{}".format(full_df.iloc[0]["commit"]), "_{}".format(per_commit_df.iloc[0]["commit"])))
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('commit')]
    return full_df


output_results_dict = main()
df = dict_to_csv(output_results_dict)
sbs_df = side_by_side_view(df)
sbs_df.to_csv("result.csv")
