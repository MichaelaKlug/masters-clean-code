import lightning as L
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData,XYSingleSquareData
from disent.dataset.sampling import GroundTruthPairOrigSampler, RlSampler, GroundTruthPairSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import AdaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.util import is_test_run  # you can ignore and remove this
from disent.metrics import metric_dci
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import torch.distributed as dist

# -----------------------------------------------------
# Argument Parser Setup
# -----------------------------------------------------
parser = argparse.ArgumentParser(description="ada-gvae with configurable hyperparameters")

parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--z_size', type=int, default=6, help='Size of latent space')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--beta', type=float, default=0.01, help='Beta')
parser.add_argument('--max_steps', type=int, default=57600, help='Number of training steps')
parser.add_argument('--title', type=str, default='', help='Run description')
parser.add_argument('--rl_sampler', type=str, default='False', help='If you are using RL sampler')

args = parser.parse_args()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def train(n_step,beta,batch_size,z_size,lr,sampler,steps):
    # prepare the data
    #data = XYSingleSquareData(grid_spacing=4,n=n_step)
    
    sampler=sampler
    data=XYObjectData()
    dataset = DisentDataset(data, sampler=sampler, transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # create the pytorch lightning system
    module: L.LightningModule = AdaVae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=z_size, z_multiplier=2),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=z_size),
        ),
        cfg=AdaVae.cfg(
            optimizer="adam",
            optimizer_kwargs=dict(lr=lr),
            loss_reduction="mean_sum",
            beta=beta,
            ada_average_mode="gvae",
            ada_thresh_mode="kl",
        ),
    )

    # train the model
    trainer = L.Trainer(max_steps=steps,logger=False, enable_checkpointing=False)
    trainer.fit(module, dataloader)
    #trainer.save_checkpoint("/home/michaela/minigrid/ada-gave/disent/disent/trained_models/trained_xyobject.ckpt")
    get_repr = lambda x: module.encode(x.to(module.device))
    metrics = {
        **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
    }
    return metrics

def write_metrics(metric,lr,batch_size,z_size,steps,description,other):
        
    # New data to add
    if not is_main_process():
        return  # only main writes

    print("[INFO] Writing metrics to CSV")

    new_row = {'inf train':metric['dci.informativeness_train'],'inf test':metric['dci.informativeness_test'],
                'disentanglment': metric['dci.disentanglement'],'completenes':metric['dci.completeness'],
                'Learning rate': lr, 'Batch size': batch_size, 'Latent size': z_size, 'Other':other,
                'Max steps': steps, 'Run description':description}
    # File path
    csv_file = 'dci_values_newest.csv'
    # Append row
    df = pd.DataFrame([new_row])
    # Append without writing the header again
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

lr=args.learning_rate
z_size=args.z_size
batch_size=args.batch_size
beta=args.beta
max_steps=args.max_steps


if args.rl_sampler=='True':
    print('RL sampling')
    # betas=[0.0001,0.001, 0.01, 0.05,1,4]
    #betas=[ 0.01, 0.05,1,4]
    #betas=[0.00001]

    steps=[0,1,20,40,60,80,100,110]
    sampler=RlSampler()
    
    for i in steps:
        metrics=train(n_step=i,beta=beta,batch_size=batch_size,z_size=z_size,lr=lr,sampler=sampler,steps=max_steps)
        print(metrics)
        title = f'xy square grid_spacing=4, RL sampler n={i}'
        print(title)
        write_metrics(metrics, lr, batch_size, z_size, max_steps, title,f'beta={beta}')

sampler=GroundTruthPairOrigSampler()
metrics=train(n_step=1,beta=beta,batch_size=batch_size,z_size=z_size,lr=lr,sampler=sampler, steps=max_steps)
print(metrics)
title = args.title
print(title)
write_metrics(metrics, lr, batch_size=batch_size, z_size=z_size, steps=max_steps, description=title,other=f'beta={beta}')
