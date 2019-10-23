import numpy as np
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', type=str, nargs='+')
    parser.add_argument('--num_0_better', type=int, default=5)
    parser.add_argument('--num_1_better', type=int, default=5)
    args = parser.parse_args()

    mses = []
    for d in args.results_dirs:
        with open(os.path.join(d, 'mses.json'), 'r') as f:
            mses.append(json.load(f))

    diffs = {}

    for k in mses[0]:
        if k in mses[1]:
            diffs[k] = float(mses[0][k]) - float(mses[1][k])

    sorted_diffs = sorted(diffs.items(), key=lambda kv: kv[1])

    print("Examples where", args.results_dirs[0], "performed better")
    for i in range(args.num_0_better):
        print(sorted_diffs[i][0], "mse:", mses[0][sorted_diffs[i][0]], "vs", mses[1][sorted_diffs[i][0]], "diff", sorted_diffs[i][1])

    print("Examples where", args.results_dirs[1], "performed better")
    for i in range(args.num_1_better):
        j = len(sorted_diffs) - 1 - i
        print(sorted_diffs[j][0], "mse:", mses[0][sorted_diffs[j][0]], "vs", mses[1][sorted_diffs[j][0]], "diff", sorted_diffs[j][1])

if __name__ == '__main__':
    main()
