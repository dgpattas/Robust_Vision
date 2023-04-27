import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
import utils
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/medium',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/medium', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./weights/medium.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()

##########################
print('****')
print('The path where the data are searched is: ', args.data_path, '\n')
print('The path where the results are saved is: ', args.save_path, '\n')
print('The model used is: ', args.model, '\n')
##########################

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)


if torch.cuda.is_available():
    if int(args.gpu) != -1:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True, 
        generator = torch.Generator(device='cuda') if int(args.gpu) != -1 else torch.Generator(device='cpu'))


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    # print('beginning\n')
    # This must be changes since we cannot allow the exit of the system if the laptop does not have the gpu
    if not torch.cuda.is_available():
        print('no gpu device available')
        # sys.exit(1)


    model = Finetunemodel(int(args.gpu), args.model)
    model = model.cuda() if int(args.gpu) != -1 else model.to(torch.device('cpu'))
    # print('model loaded\n')

    model.eval()
    # print('model in evaluation mode \n')
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda() if int(args.gpu) != -1 else  Variable(input, volatile=True).to(torch.device('cpu'))
            image_name = os.path.basename(image_name[0])
            image_name = os.path.splitext(image_name)[0]
            i, r = model(input)
            u_name = '{}.png'.format(image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)

if __name__ == '__main__':
    main()
