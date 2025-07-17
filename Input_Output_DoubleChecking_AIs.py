import struct
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
torch.backends.cudnn.enabled = True

class MNISTDataset(data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Modify value 92210 to take other probe. Keep in mind that it should divide by 10 without remainder.
        image = self.images[index + 92210]
        if index <= (len(self.images) - 65):
            image2 = self.images[index + 64]
        else:
            image2 = self.images[index]
        return torch.from_numpy(image).unsqueeze(0).float(), torch.from_numpy(image2).unsqueeze(0).float()

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)
        images = images.astype(np.float32) / 255.0
    return images

# Load the test images
test_images = load_mnist_images('shifted_mnist.bin')

# Create a dataset
dataset = MNISTDataset(test_images)
train_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Image processing component
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 30, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(30, 40, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 30, kernel_size=5),  # Changed from 20 to 40
            nn.ReLU(),
            nn.ConvTranspose2d(30, 20, kernel_size=5),  # Added this layer
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, image):
        # Image processing
        encoded_image = self.encoder(image)

        # Decode the image
        decoded_image = self.decoder(encoded_image)
        decoded_image = nn.functional.interpolate(decoded_image, size=(28, 28), mode='bilinear')

        return decoded_image

def add_salt_noise(image, p):
    noise_mask = (torch.rand_like(image) < p).to(torch.bool)
    noisy_image = image
    noisy_image[noise_mask] = 1.0
    return noisy_image

def calculate_row_averages(image):
    # image: (1, 28, 28, 1)
    row_averages = torch.mean(image, dim=3, keepdim=True)  # calculate mean along rows (dim=3)
    return row_averages

def modify_precision_leftmost_pixels(batch):
    modified_batch = batch.clone()
    row_averages = calculate_row_averages(batch)  # shape: (1, 28, 1, 1)
    modified_batch[:, :, :, 0] = row_averages[:, :, :, 0]  # remove extra dimensions
    return modified_batch

def extract_leftmost_side(batch):
    leftmost_side = batch[:, :, :, 0].unsqueeze(-1)
    return leftmost_side

def apply_leftmost_side(batch, leftmost_side):
    modified_batch = batch.clone()
    for i in range(batch.size(0)):
        modified_batch[i, :, :, 0] = leftmost_side[:, :, :, 0]
    return modified_batch

def apply_leftmost_side_cutting(batch):
    modified_batch = batch.clone()
    for i in range(batch.size(0)):
        modified_batch[i, :, :, 0] = 0.0
    return modified_batch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('model_orig_mod_weights_deadman0.105.pth', map_location=device))

    counter = 0
    for i, (image, _) in enumerate(train_loader):
        image = image.to(device)
        input_image = torch.clone(image)
#        input_image = add_salt_noise(input_image, 0.105 * counter)  # comparison condition main parameter 0.105, mode noised/denoise
        input_image = add_salt_noise(input_image, 0.105 * 9) # comparison condition main parameter 0.105, mode deadman
        print(f"Input shape: {image.shape}")
        model.load_state_dict(torch.load('model_orig_mod_weights_deadman0.105.pth', map_location=device))
        if counter == 0:
            output_prec = model(input_image)
            leftmost_side = extract_leftmost_side(output_prec)
        else:
            with torch.no_grad():
                image_applied = torch.clone(input_image)
                image_applied = apply_leftmost_side(image_applied, leftmost_side)
            output_prec = model(image_applied)
            leftmost_side = extract_leftmost_side(output_prec)
        model.load_state_dict(torch.load('model_cutside_mod_weights_deadman0.105.pth', map_location=device))
        if counter == 0:
            output_no_prec = model(input_image)
            counter += 1
        else:
            with torch.no_grad():
                other_image_applied = torch.clone(input_image)
                other_image_applied = apply_leftmost_side_cutting(image_applied) # comment this for comparison without cutting
            output_no_prec = model(other_image_applied)
            counter += 1
        print(f"Output shape: {output_prec.shape}")
        print(f"Output values: {output_prec.min().item():.4f} - {output_prec.max().item():.4f}")

        # Visualize input and output images
        original_image = image.squeeze().cpu().numpy()
        input_image = input_image.squeeze().cpu().numpy()
        if counter > 1:
            applied_image = image_applied.squeeze().cpu().numpy()
            applied_image_other = other_image_applied.squeeze().cpu().numpy()
        output_image_prec = output_prec.squeeze().cpu().detach().numpy()
        output_image_no_prec = output_no_prec.squeeze().cpu().detach().numpy()

        plt.figure(figsize=(20, 10))
        # Change strings as you wish.
        plt.subplot(1, 5, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        if counter == 1:
            plt.subplot(1, 5, 2)
            plt.imshow(input_image, cmap='gray')
            plt.title('Input Image before scope')
        elif counter > 1:
            plt.subplot(1, 5, 2)
            plt.imshow(applied_image_other, cmap='gray')
            plt.title('Input Image with cutting')
        if counter > 1:
            plt.subplot(1, 5, 3)
            plt.imshow(applied_image, cmap='gray')
            plt.title('Input Image after scope')

        plt.subplot(1, 5, 4)
        plt.imshow(output_image_prec, cmap='gray')
        plt.title('Output for model with precision')

        plt.subplot(1, 5, 5)
        plt.imshow(output_image_no_prec, cmap='gray')
        plt.title('Output for model without precision')

        directory = 'FirstAI/weights_archive/comparison_cutside-orig_deadman0.105/comparison_probe6_deadman0.105'
        filename = f'image{counter}_probe6_comparison_noised0.105.png'

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(os.path.join(directory, filename))

        plt.show()

        if i >= 9:
            break