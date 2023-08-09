#from scipy.misc import imread, imsave
import torch
from torch.autograd import Variable
from cannyedge.net_canny import Net
#import cv
import cv2

def canny(raw_img, use_cuda=False):
    #img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    img = raw_img
    
    batch = torch.stack([img]).float()

    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()
    
    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)
    #imsave('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0])
    #imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
    #imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
    #imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])

    return (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)

#if __name__ == '__main__':
def callCanny(img):
    #img = cv2.imread('img1.png') / 255.0
    #print(img1.shape)
    #img = cv2.imread('in1.png') / 255.0
    #print('imggg: ',img.shape)
    #color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    

    # canny(img, use_cuda=False)
    canny(img, use_cuda=True)
