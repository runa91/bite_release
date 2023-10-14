
import pickle as pkl
import torch


def load_dog_betas_for_3dcgmodel_loss(data_path, smal_model_type):
    assert smal_model_type in {'barc', '39dogs_diffsize', '39dogs_norm', '39dogs_norm_newv2', '39dogs_norm_newv3'}
    # load betas for the figures which were used to create the dog model
    if smal_model_type in  ['barc', '39dogs_norm', '39dogs_norm_newv2', '39dogs_norm_newv3']:
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        dog_betas_unity = data['dogs_betas']
    elif smal_model_type == '39dogs_diffsize':
        with open(data_path, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        dog_betas_unity = data['toys_betas']
    # load correspondencies between those betas and the breeds
    if smal_model_type == 'barc':
        dog_betas_for_3dcgloss = {29: torch.tensor(dog_betas_unity[0, :]).float(),
                            91: torch.tensor(dog_betas_unity[1, :]).float(),
                            84: torch.tensor(0.5*dog_betas_unity[3, :] + 0.5*dog_betas_unity[14, :]).float(),
                            85: torch.tensor(dog_betas_unity[5, :]).float(),
                            28: torch.tensor(dog_betas_unity[6, :]).float(),
                            94: torch.tensor(dog_betas_unity[7, :]).float(),
                            92: torch.tensor(dog_betas_unity[8, :]).float(),
                            95: torch.tensor(dog_betas_unity[10, :]).float(),
                            20: torch.tensor(dog_betas_unity[11, :]).float(),
                            83: torch.tensor(dog_betas_unity[12, :]).float(),
                            99: torch.tensor(dog_betas_unity[16, :]).float()}
    elif smal_model_type in ['39dogs_diffsize', '39dogs_norm', '39dogs_norm_newv2', '39dogs_norm_newv3']:
        dog_betas_for_3dcgloss = {84: torch.tensor(dog_betas_unity[0, :]).float(),
                        99: torch.tensor(dog_betas_unity[2, :]).float(),
                        81: torch.tensor(dog_betas_unity[6, :]).float(),    
                        9: torch.tensor(dog_betas_unity[9, :]).float(),
                        40: torch.tensor(dog_betas_unity[10, :]).float(),
                        29: torch.tensor(dog_betas_unity[11, :]).float(),
                        10: torch.tensor(dog_betas_unity[13, :]).float(),
                        11: torch.tensor(dog_betas_unity[14, :]).float(),
                        44: torch.tensor(dog_betas_unity[15, :]).float(),
                        91: torch.tensor(dog_betas_unity[16, :]).float(),
                        28: torch.tensor(dog_betas_unity[17, :]).float(),
                        108: torch.tensor(dog_betas_unity[20, :]).float(),
                        80: torch.tensor(dog_betas_unity[21, :]).float(), 
                        85: torch.tensor(dog_betas_unity[23, :]).float(),
                        68: torch.tensor(dog_betas_unity[24, :]).float(),
                        94: torch.tensor(dog_betas_unity[25, :]).float(),
                        95: torch.tensor(dog_betas_unity[26, :]).float(),
                        20: torch.tensor(dog_betas_unity[27, :]).float(),
                        62: torch.tensor(dog_betas_unity[28, :]).float(),
                        57: torch.tensor(dog_betas_unity[30, :]).float(),
                        102: torch.tensor(dog_betas_unity[31, :]).float(),
                        8: torch.tensor(dog_betas_unity[35, :]).float(),
                        83: torch.tensor(dog_betas_unity[36, :]).float(),
                        96: torch.tensor(dog_betas_unity[37, :]).float(),
                        46: torch.tensor(dog_betas_unity[38, :]).float()}
    return dog_betas_for_3dcgloss