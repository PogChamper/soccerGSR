#!/usr/bin/env python3

import argparse
import yaml
import fiftyone as fo


def load_config():
    """Load hub configuration."""
    with open("conf/config_ingest.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config['hub']['name']


def delete_hub(hub_name: str):
    """Delete the dataset hub if it exists."""
    if hub_name not in fo.list_datasets():
        print(f"Hub '{hub_name}' does not exist")
        return
    
    print(f"Deleting hub: {hub_name}")
    fo.delete_dataset(hub_name)
    print(f"Hub '{hub_name}' deleted successfully")


def view_hub(hub_name: str):
    """Launch FiftyOne App to view the dataset hub."""
    if hub_name not in fo.list_datasets():
        print(f"Hub '{hub_name}' not found.")
        print("Run: python ingest_hub.py")
        return
    
    print(f"Loading hub: {hub_name}")
    hub = fo.load_dataset(hub_name)
    
    print(f"Total samples: {len(hub)}")
    print("\nAvailable tags:")
    for tag in sorted(hub.distinct("tags")):
        count = len(hub.match_tags(tag))
        print(f"  {tag:25s}: {count:6d} samples")
    
    print("\n" + "=" * 80)
    print("Launching FiftyOne App...")
    print("Press Ctrl+C to exit")
    print("=" * 80)
    
    session = fo.launch_app(hub)
    session.wait()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View or manage the FiftyOne dataset hub"
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete the dataset hub instead of viewing it'
    )
    
    args = parser.parse_args()
    hub_name = load_config()
    
    if args.delete:
        delete_hub(hub_name)
    else:
        view_hub(hub_name)


if __name__ == "__main__":
    main()

