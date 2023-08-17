import argparse

# def main():
parser = argparse.ArgumentParser(description="Basic CLI using argparse.")
parser.add_argument("--source", "-s", type=str, required=True, help="Source file or directory.")
parser.add_argument("--target", "-t", type=str, required=True, help="Target file or directory.")
args = parser.parse_args()

print(f"Source: {args.source}")
print(f"Target: {args.target}")