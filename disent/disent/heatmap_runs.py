import lightning as L
from torch.utils.data import DataLoader
import numpy as np
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

import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def train(beta,data,dataloader):
    # prepare the data
    #data = XYSingleSquareData(grid_spacing=4,n=n_step)
    
    
    # create the pytorch lightning system
    module: L.LightningModule = AdaVae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=data.x_shape, z_size=9, z_multiplier=2),
            decoder=DecoderConv64(x_shape=data.x_shape, z_size=9),
        ),
        cfg=AdaVae.cfg(
            optimizer="adam",
            optimizer_kwargs=dict(lr=1e-3),
            loss_reduction="mean_sum",
            beta=beta,
            ada_average_mode="gvae",
            ada_thresh_mode="kl",
        ),
    )

    # train the model
    trainer = L.Trainer(max_steps=57600,
                        logger=False, 
                        enable_checkpointing=False,
                        devices=1,
                        )
    trainer.fit(module, dataloader)
    #trainer.save_checkpoint("/home/michaela/minigrid/ada-gave/disent/disent/trained_models/trained_xyobject.ckpt")
    # get_repr = lambda x: module.encode(x.to(module.device))
    # metrics = {
    #     **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
    # }
    # return metrics

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
    csv_file = 'dci_values_rl.csv'
    # Append row
    df = pd.DataFrame([new_row])
    # Append without writing the header again
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)




#steps=[0,1,20,40,60,80,100,110]
# betas=[0.0001,0.001, 0.01, 0.05,1,4]
steps=[40]
# for beta in betas:
beta=0.01
for i in steps:
    data = XYSingleSquareData(grid_spacing=4,n=i)
    dataset = DisentDataset(data, RlSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
    train(beta=beta,data=data,dataloader=dataloader)
    # print(metrics)
    # title = f'xy square grid_spacing=4, beta {beta}, RL sampler n={i}'
    # print(title)
    # write_metrics(metrics, 1e-3, 4, 9, 57600, title,f'beta={beta}')
    
    #plt.imsave(f'heat_images/square_spacing4_beta_{beta}_n_{i}.png', data.accum)
    #heatmap = data.accum / len(dataloader.dataset)  # Normalize
    # heatmap=data.accum_start
    # plt.imshow(heatmap, cmap='hot')
    # plt.colorbar()
    # plt.savefig(f'heat-maps-start-pair/n_{i}_start.png')
    # plt.close() 

    # heatmap2=data.accum_pair
    # plt.imshow(heatmap2, cmap='hot')
    # plt.colorbar()
    # plt.savefig(f'heat-maps-start-pair/n_{i}_pair.png')
    # plt.close()
    # 
    plt.figure()
    plt.imshow(data.accum_start, cmap='hot')#, vmin=np.min(data.accum_start), vmax=np.max(data.accum_start))
    plt.colorbar()
    plt.savefig(f'heat-maps-start-pair/square_spacing4_beta_{beta}_n_{i}_start4.png')
    plt.close()

    plt.figure()
    plt.imshow(data.accum_pair, cmap='hot')#, vmin=np.min(data.accum_pair), vmax=np.max(data.accum_pair))
    plt.colorbar()
    plt.savefig(f'heat-maps-start-pair/square_spacing4_beta_{beta}_n_{i}_pair4.png')
    plt.close() 

    with open("counts.txt", "a") as f:
        f.write(f"pair count = {data.count_pair}\n")
        f.write(f"start count = {data.count_start}\n")

# metrics=train(n_step=1,beta=4)
# print(metrics)
# title = f'xy object, orig sampler'
# write_metrics(metrics, 1e-3, 64, 9, 57600, title,'beta=4')
# data = XYSingleSquareData(grid_spacing=4,n=1)
# dataset = DisentDataset(data, GroundTruthPairOrigSampler(), transform=ToImgTensorF32(),return_indices=True)
# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
# train(beta=beta,data=data,dataloader=dataloader)
# metrics=train(beta=beta,data=data,dataloader=dataloader)
# print(metrics)
# title = f'xy square grid_spacing=4, beta {beta}, orig pair sampler'
# print(title)
# write_metrics(metrics, 1e-3, 4, 9, 57600, title,f'beta={beta}')

#plt.imsave(f'heat_images/square_spacing4_beta_{beta}_n_{i}.png', data.accum)
#heatmap = data.accum / len(dataloader.dataset)  # Normalize
# heatmap=data.accum_start
# plt.imshow(heatmap, cmap='hot')
# plt.colorbar()
# plt.show()
# plt.savefig(f'heat-maps-start-pair/origpair_start.png')
# plt.close() 

# heatmap2=data.accum_pair
# plt.imshow(heatmap2, cmap='hot')
# plt.colorbar()
# plt.show()
# plt.savefig(f'heat-maps-start-pair/origpair_pair.png')
# plt.close() 

# with open("counts.txt", "a") as f:
#     f.write(f"{data.count}\n")