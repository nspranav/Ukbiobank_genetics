#!/bin/sh
conda activate latest3
python lightning_classification.py --load_from 5216125 --group L2+L3
python lightning_classification.py --load_from 5219228 --group L2+L4
python lightning_classification.py --load_from 5215591 --group L3+L4
python lightning_classification.py --load_from 5219373 --group L1+L2+L3
python lightning_classification.py --load_from 5219628 --group L1+L2+L3+L4

