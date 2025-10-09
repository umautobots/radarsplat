import os
import argparse

def get_sorted_timestamps(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    timestamps = [int(os.path.splitext(f)[0]) for f in image_files]
    sorted_timestamps = sorted(timestamps)
    return sorted_timestamps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder with .png images")
    parser.add_argument("--timestamp", type=int, help="Timestamp to find index for")
    parser.add_argument("--index", type=int, help="Index to get timestamp at")
    args = parser.parse_args()

    sorted_ts = get_sorted_timestamps(args.folder)

    if args.timestamp is not None:
        try:
            idx = sorted_ts.index(args.timestamp)
            print(f"Timestamp {args.timestamp} is at index {idx}.")
        except ValueError:
            print(f"Timestamp {args.timestamp} not found in the folder.")

    if args.index is not None:
        if 0 <= args.index < len(sorted_ts):
            print(f"Index {args.index} corresponds to timestamp {sorted_ts[args.index]}.")
        else:
            print(f"Index {args.index} is out of range (0 to {len(sorted_ts) - 1}).")

    if args.timestamp is None and args.index is None:
        print("Sorted timestamps:")
        for i, ts in enumerate(sorted_ts):
            print(f"{i}: {ts}")

if __name__ == "__main__":
    main()