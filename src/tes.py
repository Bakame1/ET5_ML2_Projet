import torch
print(torch.cuda.is_available())        # Doit être True
print(torch.cuda.get_device_name(0))   # Doit afficher le nom de ton GPU