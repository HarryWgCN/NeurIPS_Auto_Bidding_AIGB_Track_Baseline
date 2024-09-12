# import pandas as pd
#
# df = pd.read_csv('/home/disk2/auto-bidding/data/traffic/period-7.csv')
# print(len(df))
# df = df[:1000000]
# df.to_csv('/home/disk2/auto-bidding/data/truncated/period-7-truncated.csv')
# print()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

print('bye')
logger.info('hi')
