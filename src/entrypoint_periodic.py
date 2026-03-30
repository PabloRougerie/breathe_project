from src.flows.periodic import periodic_monitoring_masterflow
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_num", type=int, required=False, default=None)
    args = parser.parse_args()
    periodic_monitoring_masterflow(batch_num=args.batch_num)
