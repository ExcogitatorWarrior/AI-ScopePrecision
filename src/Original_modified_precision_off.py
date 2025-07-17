import struct
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.optim as optim


class MNISTDataset(data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        # Probably not a most computationally effective way to achieve batch training.
        # You might want to pay attention to it and change for different batches.
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

def add_salt_noise(image, image2, p):
    noise_mask = (torch.rand_like(image) < p).to(torch.bool)
    noisy_image = image
    noisy_image2 = image2
    noisy_image[noise_mask] = 1.0
    noisy_image2[noise_mask] = 1.0
    return noisy_image, noisy_image2

def add_salt_noise_single(image, p):
    noise_mask = (torch.rand_like(image) < p).to(torch.bool)
    noisy_image = image
    noisy_image[noise_mask] = 1.0
    return noisy_image

def calculate_row_averages(image):
    row_averages = torch.mean(image, dim=3, keepdim=True)
    return row_averages

def modify_precision_leftmost_pixels(batch, original_batch):
    modified_batch = batch.clone()
    support_batch = original_batch.clone()
    row_averages = calculate_row_averages(support_batch)
    modified_batch[:, :, :, 0] = row_averages[:, :, :, 0]
    return modified_batch

def extract_leftmost_side(batch):
    leftmost_side = batch[:, :, :, 0].unsqueeze(-1)
    return leftmost_side

def apply_leftmost_side(batch, leftmost_side):
    modified_batch = batch.clone()
    for i in range(batch.size(0)):
        modified_batch[i, :, :, 0] = leftmost_side[i].squeeze() + 0.0
    return modified_batch

# Load the test images
test_images = load_mnist_images('shifted_mnist_64_batch.bin')

# Create a dataset
dataset = MNISTDataset(test_images)
train_loader = data.DataLoader(dataset, batch_size=64, shuffle=False)

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
            nn.ConvTranspose2d(40, 30, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(30, 20, kernel_size=5),
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

def loss_function(decoded_image, image):
    # Compute the loss
    loss = nn.MSELoss()(decoded_image, image)
    return loss

def train_model(device, model, loader, optimizer, num_epochs):
    model.to(device)
    losses = []
    for epoch in range(num_epochs):
        counter = 0
        for image, image2 in loader:
            input_image, compare_image = torch.clone(image), torch.clone(image2)
#            input_image = add_salt_noise_single(input_image, 0.09 * 9)  # This code used for deadman mode. 0.09 is main parameter
#            input_image = add_salt_noise_single(input_image, 0.11 * counter) # this code used for denoise mode. 0.11 is main parameter
            input_image, compare_image = add_salt_noise(input_image, compare_image, 0.105 * counter) # this code used for noised mode. 0.105 is main parameter
            input_image, compare_image  = input_image.to(device), compare_image.to(device)
            if counter == 0:
                decoded_image = model(input_image)
            elif counter < 9:
                decoded_image = model(input_image)
            if counter < 9:
                loss = loss_function(decoded_image, compare_image)
                counter += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            else:
                counter = 0
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    return losses

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 100
#    model.load_state_dict(torch.load('model_orig_mod_weights_noised.pth', map_location=device))

    losses = train_model(device, model, train_loader, optimizer, num_epochs)

    torch.save(model.state_dict(), 'model_no_precision_mod_weights_noised0.105.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('No Precision noised 0.105 Loss Over Time')
    plt.show()