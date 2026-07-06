"""Train a pyramid RVQ codec.

  python scripts/train_pyramid.py -c configs/pyr64.yaml --gpu 0 \
      log.run_tag=my_run [key.path=value ...]
"""
from georvq.cli import train_pyramid

if __name__ == "__main__":
    train_pyramid()
