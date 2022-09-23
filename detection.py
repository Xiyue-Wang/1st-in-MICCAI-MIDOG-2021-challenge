import os
import logging
import torch
from queue import Queue, Empty
from tqdm import tqdm
import numpy as np
from skimage import measure
import cv2
import torch.nn as nn
from PIL import Image
import gc

from model.model import FMDet


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


seed_torch(2021)


class UNetMitosisDetection:
    def __init__(self, path_model, size, batchsize, out_threshold):
        ## network parameters
        # TODO: The new model architecture
        self.out_thresh = out_threshold
        self.path_model = path_model
        self.size = size
        self.batchsize = batchsize
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        if torch.cuda.is_available():
            print("Model loaded on CUDA")
            net1=FMDet()
            net1 = torch.nn.DataParallel(net1).cuda()
            net1.load_state_dict(torch.load(self.path_model['net1']))
            net1.eval()



            self.model = [net1]

        logging.info("Model loaded.")


    def process_image(self, input_image):
        n_patches = 0
        queue_patches = Queue()
        img_dimensions = input_image.shape

        # create overlapping patches for the whole image       
        for x in np.arange(0, img_dimensions[1], int(0.5 * self.size)):
            for y in np.arange(0, img_dimensions[0], int(0.5 * self.size)):
                # last patch shall reach just up to the last pixel
                if (x+self.size>img_dimensions[1]):
                    x = img_dimensions[1]-self.size

                if (y+self.size>img_dimensions[0]):
                    y = img_dimensions[0]-self.size

                queue_patches.put((0, int(x), int(y), input_image))
                n_patches += 1

        # define an empty mask
        mask_slide=np.zeros(img_dimensions[:2],dtype=np.float16)
        mask_num_adding = np.zeros(img_dimensions[:2],dtype=np.float16)
        n_batches = int(np.ceil(n_patches / self.batchsize))

        for _ in tqdm(range(n_batches), desc='Processing an image'):
            torch_batch, batch_x, batch_y = self.get_batch(queue_patches)
            pred_seg = None
            with torch.no_grad():
                for net in self.model:

                    net.eval()
                    # TTA
                    mask_preds = net(torch_batch)
                    output0 = torch.sigmoid(mask_preds)

                    output1 = net(torch_batch.flip(3))
                    output1 = torch.sigmoid(output1.flip(3))

                    output2 = net(torch_batch.flip(2))
                    output2 = torch.sigmoid(output2.flip(2))

                    mask_preds=(output0+output1+output2)/3.0
                    mask_preds=mask_preds.cpu().numpy()

                    if pred_seg is None:
                        pred_seg = mask_preds
                    else:
                        pred_seg += mask_preds

                pred_seg /= len(self.model)


            for b in range(torch_batch.shape[0]):
                x_real = batch_x[b]
                y_real = batch_y[b]
                mask_slide[y_real:y_real+self.size, x_real:x_real+self.size] += pred_seg[b,0,:,:]
                mask_num_adding[y_real:y_real+self.size, x_real:x_real+self.size] += 1

        mask_num_adding[mask_num_adding==0]=1
        mask_slide = mask_slide / mask_num_adding
        mask_slide = (mask_slide > self.out_thresh).astype(np.uint8)

        mask_slide = cv2.medianBlur(mask_slide, 3)
        kernel = np.ones((20, 20), np.uint8)
        mask_slide = cv2.morphologyEx(mask_slide, cv2.MORPH_CLOSE, kernel)

        slide_boxes = self.convert_to_bbox(mask_slide)

        del mask_slide, mask_num_adding
        gc.collect()

        print(slide_boxes)

        return slide_boxes

    def get_batch(self, queue_patches):
        batch_images = np.zeros((self.batchsize, 3, self.size, self.size))
        batch_x = np.zeros(self.batchsize, dtype=int)
        batch_y = np.zeros(self.batchsize, dtype=int)
        for i_batch in range(self.batchsize):
            if queue_patches.qsize() > 0:
                status, batch_x[i_batch], batch_y[i_batch], image = queue_patches.get()
                x_start, y_start = int(batch_x[i_batch]), int(batch_y[i_batch])
                
                if image.shape[-1] == 3:
                    img_pil_rgb = Image.fromarray(image, mode='RGB')
                elif image.shape[-1] == 4:
                    img_pil_rgb = Image.fromarray(image, mode='RGBA')   
                    img_pil_rgb = img_pil_rgb.convert('RGB')

                img_pil_rgb = np.array(img_pil_rgb) # (HWC)
                cur_patch = img_pil_rgb[y_start:y_start+self.size, x_start:x_start+self.size] / 255.
                batch_images[i_batch] = cur_patch[:,:,::-1].transpose(2, 0, 1)
            else:
                batch_images = batch_images[:i_batch]
                batch_x = batch_x[:i_batch]
                batch_y = batch_y[:i_batch]
                break
        torch_batch = torch.from_numpy(batch_images.astype(np.float32, copy=False)).to(self.device)
        return torch_batch, batch_x, batch_y


    def convert_to_bbox(self, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_labeled = measure.label(mask, connectivity=2)
        
        num_bbox = mask_labeled.max()
        bboxes = []
        
        h, w = mask.shape[0], mask.shape[1]

        if num_bbox >= 200:
            return np.array(bboxes)

        for i in range(1, num_bbox+1):
            mask_per_inst = (mask_labeled == i).astype(np.uint8)
            xs, ys = np.where(mask_per_inst==1)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            if num_bbox < 20:
                if (0 in xs) or  (h-1 in xs) or (0 in ys) or (w-1 in ys):
                    # print(slide, [y1, x1, y2, x2])
                    # print('Filter one box connected to boundary')
                    continue
                else:
                    bboxes.append(np.array([y1, x1, y2, x2]))
            else:
                bboxes.append(np.array([y1, x1, y2, x2]))
        return np.array(bboxes)   