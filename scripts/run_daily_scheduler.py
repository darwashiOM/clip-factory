from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler_core import schedule_next_days_core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--posts-per-day", type=int, default=10)
    parser.add_argument("--start-hour", type=int, default=8)
    parser.add_argument("--end-hour", type=int, default=22)
    parser.add_argument("--tag", default="")
    parser.add_argument("--platform", default="tiktok")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = schedule_next_days_core(
        account=args.account,
        days=args.days,
        posts_per_day=args.posts_per_day,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        tag=args.tag,
        platform=args.platform,
        dry_run=args.dry_run,
    )
    print(result)


if __name__ == "__main__":
    main()
