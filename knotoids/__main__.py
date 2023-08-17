import argparse

# def main():
parser = argparse.ArgumentParser(description="Computes the knotoid distribution of a 3D PL curve.")
parser.add_argument("--source", "-s", type=str, required=True, help="Source file of PL curve coordinates.")
parser.add_argument("--target", "-t", type=str, required=True, help="Target file.")
args = parser.parse_args()

print(f"Source: {args.source}")
print(f"Target: {args.target}")