import argparse
import os
from src.vehicle_reid.config import load_config
from src.vehicle_reid.engine.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.project.output_dir, exist_ok=True)
    os.makedirs(cfg.project.weights_dir, exist_ok=True)

    run_training(cfg)


if __name__ == "__main__":
    main()
