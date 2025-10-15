import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-r', '--ratio', type=float, default=0.3, help="compression ratio (= retained/full)")
parser.add_argument(
    '--kv_type',
    type=str,
    default='evict',
    choices=['evict', 'retain'],
    help="retain: full cache in storage for effcient evaluation over multiple compression ratios")
parser.add_argument(
    '--level',
    type=str,
    default='pair',
    choices=['pair', 'head', 'pair-uniform'],
    help="head: context-independent head-level eviction. pair-uniform: uniform head-budget ratios")

parser.add_argument(
    '-m',
    '--model',
    type=str,
    help=
    "check the model list in model/load.py. recommended to use abbreviated model names, e.g., llama3.1-8b, qwen2.5-7b"
)

parser.add_argument('-d',
                    '--data',
                    type=str,
                    help="check the dataset list in data/load.py (e.g., squad, needle, scbench_kv)")
parser.add_argument('--idx', type=int, default=0, help="the index of a data example")
parser.add_argument('--num', type=int, default=1, help="the total number of eval data")
parser.add_argument('--tag', type=str, default=None, help="evaluation folder name tag")

parser.add_argument('--save_head_score', action="store_true", help="save head-level importance score")
args = parser.parse_args()
