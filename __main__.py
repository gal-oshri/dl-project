"""Allow running as: python -m dl_project {train,infer}"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "infer"):
        print("Usage: python -m dl_project {train,infer} [OPTIONS]")
        print("\nCommands:")
        print("  train   Train the CCM compression adapter")
        print("  infer   Run inference with a trained model")
        sys.exit(1)

    command = sys.argv.pop(1)

    if command == "train":
        from dl_project.train import main as train_main
        train_main()
    elif command == "infer":
        from dl_project.infer import main as infer_main
        infer_main()


if __name__ == "__main__":
    main()
