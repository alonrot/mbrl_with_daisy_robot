import torch
import sys
import hydra
import pdb
import os

def save_tensor(cfg):
    """
    From https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

    >>> torch.device('cuda:0')
    device(type='cuda', index=0)

    >>> torch.device('cpu')
    device(type='cpu')

    >>> torch.device('cuda')  # current cuda device
    device(type='cuda')
    """

    # cfg.device is a string
    torch_device = torch.device(cfg.device)
    print("Current device:",torch_device)

    try:
        # my_tensor = torch.zeros((10,10),device=torch_device)
        my_tensor = torch.zeros((10,10),device="cuda") # Equivalent to the above
    except:
        my_tensor = torch.zeros((10,10),device="cpu") # Equivalent to the above

    my_tensor_cpu = my_tensor.to("cpu")

    # Save:
    # path = cfg.debug_cpu_cuda.path2save
    # path = os.getcwd()
    # path = "/Users/alonrot/PycharmProjects/mbrlDaisy/mbrl/"
    path = cfg.debug_cpu_cuda.path_base
    # pdb.set_trace()
    file_name_cpu = "{0:s}_cpu".format(cfg.debug_cpu_cuda.name_base)
    path_full_cpu = "{0:s}/{1:s}".format(path,file_name_cpu)
    file_name_cuda = "{0:s}_cuda".format(cfg.debug_cpu_cuda.name_base)
    path_full_cuda = "{0:s}/{1:s}".format(path,file_name_cuda)
    pdb.set_trace()
    torch.save(my_tensor,path_full_cuda)
    torch.save(my_tensor_cpu,path_full_cpu)

def load_tensor(cfg):

    # cfg.device is a string
    torch_device = torch.device(cfg.device)
    print("Current device:",torch_device)

    path = cfg.debug_cpu_cuda.path_base
    file_name_cpu = "{0:s}_cpu".format(cfg.debug_cpu_cuda.name_base)
    path_full_cpu = "{0:s}/{1:s}".format(path,file_name_cpu)
    file_name_cuda = "{0:s}_cuda".format(cfg.debug_cpu_cuda.name_base)
    path_full_cuda = "{0:s}/{1:s}".format(path,file_name_cuda)
    
    my_tensor_cpu = torch.load(path_full_cpu,map_location=torch.device(torch_device))
    my_tensor_cuda = torch.load(path_full_cuda,map_location=torch.device(torch_device))

    print(my_tensor_cpu)
    print(my_tensor_cuda)

@hydra.main(config_path='pets/conf/config.yaml', strict=True)
def main(cfg):

    if cfg.debug_cpu_cuda.what2do == "save":
        save_tensor(cfg)
    elif cfg.debug_cpu_cuda.what2do == "load":
        load_tensor(cfg)
    else:
        raise ValueError


if __name__ == "__main__":

    sys.exit(main())

    