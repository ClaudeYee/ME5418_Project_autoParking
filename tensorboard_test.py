import numpy as np
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./path/to/log', purge_step=90)

for x in range(100):
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data group', {'y=x*sin(x)': x * np.sin(x),
                                      'y=x*cos(x)': x * np.cos(x)}, x)

writer.flush()

# tensorboard --logdir=./path/to/log --port 8123