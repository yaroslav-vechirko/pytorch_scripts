'''
Trains a PyTorch image classification model using device-agnostic code.
'''
import os
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser(prog='train', description='Change hyperparameters of model.')
parser.add_argument('--train_dir', default='data/pizza_steak_sushi/train', type=str)
parser.add_argument('--test_dir', default='data/pizza_steak_sushi/test', type=str)
parser.add_argument('--num_epochs', default=5, type=int)
parser.add_argument('--batch_size',  default=32, type=int)
parser.add_argument('--hidden_units', default=10, type=int)
parser.add_argument('--learning_rate', default=0.001, type=int)
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
train_dir = args.train_dir
test_dir = args.test_dir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size=BATCH_SIZE)

model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=HIDDEN_UNITS,
                                output_shape=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

start_time = timer()

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


utils.save_model(model=model,
                 target_dir='models',
                 model_name='05_going_modular_script_mode_tinyvgg_model.pth')
