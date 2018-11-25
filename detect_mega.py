from network import PNet,ONet
import torch,cv2,itertools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os

use_gpu = True

def get_anchors(scale=64):
    '''
    compute anchors
    return:
        u_boxes:tensor([anchor_num,4]) (cx,cy,w,h): real anchors
        boxes:tensor([anchor_num,4]) (x1,y1,x2,y2): crop box for ONet,each with size 80
    '''
    sizes = [float(s) / scale for s in [32]]
    
    aspect_ratios = [(1.,)]
    feature_map_sizes = [int(scale/16)]
    
    num_layers = len(feature_map_sizes)
    u_boxes,boxes = [],[]
    for i in range(num_layers):
        fmsize = feature_map_sizes[i]
        for h,w in itertools.product(range(fmsize),repeat=2):
            cx = float(w)/feature_map_sizes[i]
            cy = float(h)/feature_map_sizes[i]
            
            s = sizes[i]
            for j,ar in enumerate(aspect_ratios[i]):
                u_boxes.append((cx,cy,float(s)*ar,float(s)*ar))
                boxes.append((w*16-32,h*16-32,w*16+32,h*16+32))       
    return torch.Tensor(u_boxes),torch.Tensor(boxes).long()

def nms(bboxes,scores,threshold=0.35):
    '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i) 

        if order.numel() == 1:
            break 

        xx1 = x1[order[1:]].clamp(min=x1[i]) 
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter) 
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1] 
    return torch.LongTensor(keep)
    
def decode_box(loc,size=64):
    variances = [0.1,0.2]
    anchor,crop = get_anchors(scale=size)
    cxcy = loc[:,:2] * variances[0] * anchor[:,2:] + anchor[:,:2]
    wh = torch.exp(loc[:,2:] * variances[1]) * anchor[:,2:]
    boxes = torch.cat([cxcy-wh/2,cxcy+wh/2],1)
    
    return boxes,anchor,crop
    
def decode_ldmk(ldmk,anchor):
    variances = [0.1,0.2]
    index_x = torch.Tensor([0,2,4,6,8]).long()
    index_y = torch.Tensor([1,3,5,7,9]).long()
    ldmk[:,index_x] = ldmk[:,index_x] * variances[0] * anchor[:,2].view(-1,1) + anchor[:,0].view(-1,1)
    ldmk[:,index_y] = ldmk[:,index_y] * variances[0] * anchor[:,3].view(-1,1) + anchor[:,1].view(-1,1)
    return ldmk
    
    
def detect(im,pnet,onet):
    if im is None:
        print("can not open image:", file)
        return

    # pad img to square
    h,w,_ = im.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff //2, dim_diff - dim_diff // 2
    pad = ((pad1,pad2),(0,0),(0,0)) if h<=w else ((0,0),(pad1,pad2),(0,0))
    img = np.pad(im,pad,'constant',constant_values=128)
    
    #get img_pyramid
    img_scale,img_size = 0,int((img.shape[0]-1)/32)
    while img_size > 0:
        img_scale += 1
        img_size /= 2
        if img_scale == 5:
            break
    img_size = 32
    img_pyramid = []
    t_boxes,t_probs,t_anchors,t_crops,t_which = None,None,None,None,None
    
    for scale in range(img_scale+1):
        input_img = cv2.resize(img,(img_size,img_size))
        img_pyramid.append(input_img.transpose(2,0,1))
        im_tensor = torch.from_numpy(input_img.transpose(2,0,1)).float()
        if use_gpu:
            im_tensor = im_tensor.cuda()
        
        #get conf and loc(box)
        if use_gpu:
            torch.cuda.synchronize()
        loc,conf = pnet(torch.unsqueeze(im_tensor,0))
        if use_gpu:
            torch.cuda.synchronize()     
        loc,conf = loc.detach().cpu(),conf.detach().cpu()
        loc,conf=loc.data.squeeze(0),F.softmax(conf.squeeze(0))
        boxes,anchor,crop = decode_box(loc,size=img_size)
        which_img = torch.tensor([scale]).long().expand((crop.shape[0],))
        
        #add box into stack
        if scale == 0:
            t_boxes,t_confs,t_anchors,t_crops,t_which = boxes,conf,anchor,crop,which_img
        else:
            t_boxes = torch.cat((t_boxes,boxes),0)
            t_confs = torch.cat((t_confs,conf),0)
            t_anchors = torch.cat((t_anchors,anchor),0)
            t_crops = torch.cat((t_crops,crop),0)
            t_which = torch.cat((t_which,which_img),0)
        img_size *= 2

    #get right boxes and nms
    t_confs[:,0] = 0.5
    max_conf,labels = t_confs.max(1)
    if labels.long().sum().item() is 0:
        return [],[]
    ids = labels.nonzero().squeeze(1)
    t_boxes,t_confs,t_anchors,t_crops,t_which = t_boxes[ids],t_confs[ids],t_anchors[ids],t_crops[ids],t_which[ids]
    max_conf = max_conf[ids]
    
    keep = nms(t_boxes,max_conf)
    t_boxes,max_conf,t_anchors,t_crops,t_which = t_boxes[keep],max_conf[keep],t_anchors[keep],t_crops[keep],t_which[keep]

    t_boxes = t_boxes.detach().numpy()
    max_conf = max_conf.detach().numpy()
    
    #get crop and ldmks
    crop_imgs = []
    for i in xrange(t_boxes.shape[0]):
        img = img_pyramid[t_which[i]]
        crop = t_crops[i].numpy()
        _,h_,w_ = img.shape
        o_x1,o_y1,o_x2,o_y2 = max(crop[0],0),max(crop[1],0),min(crop[2],w_),min(crop[3],h_)
        c_x1 = 0 if crop[0] >=0 else -crop[0]
        c_y1 = 0 if crop[1] >=0 else -crop[1]
        c_x2 = 64 if crop[2] <= w_ else 64 - (crop[2] - w_)
        c_y2 = 64 if crop[3] <= h_ else 64 - (crop[3] - h_)
        crop_img = np.ones((3,64,64))*128
        np.copyto(crop_img[:,c_y1:c_y2,c_x1:c_x2],img[:,o_y1:o_y2,o_x1:o_x2])
        crop_imgs.append(crop_img)
    crop_imgs = torch.from_numpy(np.array(crop_imgs)).float()
    if use_gpu:
        crop_imgs = crop_imgs.cuda()
    t_ldmks = onet(crop_imgs).detach().cpu()[:,10,:].squeeze(1)
    t_ldmks = decode_ldmk(t_ldmks,t_anchors).numpy()
    
    def change(boxes,ldmks,h,w,pad1):
        index_x = torch.Tensor([0,2,4,6,8]).long()
        index_y = torch.Tensor([1,3,5,7,9]).long()
        if h <= w:
            boxes[:,1] = boxes[:,1]*w-pad1
            boxes[:,3] = boxes[:,3]*w-pad1
            boxes[:,0] = boxes[:,0]*w
            boxes[:,2] = boxes[:,2]*w  
            ldmks[:,index_x] = ldmks[:,index_x] * w
            ldmks[:,index_y] = ldmks[:,index_y] * w - pad1
        else:
            boxes[:,1] = boxes[:,1]*h
            boxes[:,3] = boxes[:,3]*h
            boxes[:,0] = boxes[:,0]*h-pad1
            boxes[:,2] = boxes[:,2]*h-pad1
            ldmks[:,index_x] = ldmks[:,index_x] * h - pad1
            ldmks[:,index_y] = ldmks[:,index_y] * h 
        return boxes,ldmks
    t_boxes,t_ldmks = change(t_boxes,t_ldmks,h,w,pad1)
    for i in xrange(len(t_boxes)):
        box,prob,ldmk = t_boxes[i],max_conf[i],t_ldmks[i]
        ldmk = ldmk.reshape(5,2)
        if prob == 0.5:
            continue
        #print('i', i, 'prob',prob,'box', box)
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        cv2.rectangle(im,(x1,y1+4),(x2,y2),(0,255,0),2)
        cv2.putText(im, str(int(prob*100)/100.0), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255))
        for k in range(ldmk.shape[0]):
            cv2.circle(im, (ldmk[k,0],ldmk[k,1]) , 3, (0,255,0), -1)    
       
    return t_boxes,t_ldmks     

if __name__ == '__main__':
    pnet,onet = PNet(),ONet() 
    pnet.load_state_dict(torch.load('weight/msos_pnet_2_299.pt')) 
    onet.load_state_dict(torch.load('weight/msos_onet_2_299.pt'))   

    pnet.eval()
    onet.eval()
    
    if use_gpu:
        torch.cuda.set_device(3)
        pnet.cuda()
        onet.cuda()
    else:
        torch.set_num_threads(1)
        
    def mkdirP(path):
        if not os.path.exists(path):
            os.mkdir(path)

    dir = '/raid/face_dataset/megaface/facescrub'
    raw_dir = os.path.join(dir,'raw')
    pts_dir = os.path.join(dir,'pts_rsa')
    out_dir = os.path.join(dir,'outlier')
    mkdirP(pts_dir)
    mkdirP(out_dir)
    cnt = 0
    class_list = os.listdir(raw_dir)
    class_list.sort()
    for class_str in class_list:
        cnt += 1
        print(cnt)
        c_path = os.path.join(raw_dir,class_str)
        o_c_path = os.path.join(pts_dir,class_str)
        ol_c_path = os.path.join(out_dir,class_str)
        mkdirP(o_c_path)
        mkdirP(ol_c_path)
        for subfile in os.listdir(c_path):
            img_path = os.path.join(c_path,subfile)
            pts_path = os.path.join(o_c_path,subfile[:-4]+'.txt')
            if os.path.exists(pts_path):
                continue
            img = cv2.imread(img_path)
            bboxes, pts = detect(img,pnet,onet)
            if len(pts) == 0:
                print('not found 2')
                cv2.imwrite(os.path.join(ol_c_path,subfile),img)
                continue
            bboxes = bboxes.astype(np.int)
            pts = pts.astype(np.int)


            min_dis = 10000000
            select = -1
            for i in xrange(len(pts)):
                mc = pts[i].reshape(5,2)[2]
                h = img.shape[0]
                w = img.shape[1]
                #print(mc,h,w)
                dis = np.dot(mc-np.array([h/2.0,w/2.0]),mc-np.array([h/2,w/2]))
                #print(dis)
                if dis < min_dis:
                    min_dis = dis
                    select = i
            ptsi = pts[select].reshape(-1)
            np.savetxt(pts_path,ptsi)