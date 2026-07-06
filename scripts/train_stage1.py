"""Train a stage-1 continuous sigma-conditioned pyramidal autoencoder.

  python scripts/train_stage1.py -c configs/stage1.yaml --gpu 0 \
      log.run_tag=my_s1 [key.path=value ...]
"""
from georvq.cli import train_stage1

if __name__ == "__main__":
    train_stage1()
