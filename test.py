import argparse
import numpy as np
import cv2
from tqdm import tqdm
import os

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils

def test(model, dataset='cocoVal', logger=print):    
    if dataset == 'OCHumanVal':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json'
    elif dataset == 'OCHumanTest':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json'
    elif dataset == 'cocoVal':
        ImageRoot = './data/coco2017/val2017'
        AnnoFile = './data/coco2017/annotations/person_keypoints_val2017_pose2seg.json'
    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    model.eval()
    
    results_segm = []
    imgIds = []
    counter = 0
    # print("get mask result >>>")
    # mask_info = open("maskinfo.txt", "w")
    for i in tqdm(range(len(datainfos))):
        rawdata = datainfos[i]
        img = rawdata['data']
        image_id = rawdata['id']
        
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata['segms']
        # print(gt_segms)
        gt_masks = np.array([])
        # gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
            
        output = model([img], [gt_kpts], [gt_masks])

        # 一张图片上可能因为有多个annotation，所以就有多个mask
        for mask in output[0]:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if mask[i][j] == 1:
                        img[i][j] = [0, 0, 255]
            maskencode = maskUtils.encode(np.asfortranarray(mask))
            maskencode['counts'] = maskencode['counts'].decode('ascii')
            results_segm.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "score": 1.0,
                    "segmentation": maskencode
                })
        if not os.path.exists("output/"):
            os.makedirs("output/")
        cv2.imwrite(os.path.join("output/", str(image_id) + ".png"), img)
        imgIds.append(image_id)
    
    def do_eval_coco(image_ids, coco, results, flag):
        from pycocotools.cocoeval import COCOeval
        assert flag in ['bbox', 'segm', 'keypoints']
        # Evaluate
        coco_results = coco.loadRes(results)
        cocoEval = COCOeval(coco, coco_results, flag)
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() 
        return cocoEval
    
    ## 这里做的是evaluation
    # cocoEval = do_eval_coco(imgIds, datainfos.COCO, results_segm, 'segm')
    # logger('[POSE2SEG]          AP|.5|.75| S| M| L|    AR|.5|.75| S| M| L|')
    # _str = '[segm_score] %s '%dataset
    # for value in cocoEval.stats.tolist():
    #     _str += '%.3f '%value
    # logger(_str)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg Testing")
    parser.add_argument(
        "--weights",
        help="path to .pkl model weight",
        type=str,
    )
    parser.add_argument(
        "--coco",
        help="Do test on COCOPersons val set",
        action="store_true",
    )
    parser.add_argument(
        "--OCHuman",
        help="Do test on OCHuman val&test set",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(args.weights)
            
    print('===========>   testing    <===========')
    if args.coco:
        test(model, dataset='cocoVal') 
    if args.OCHuman:
        test(model, dataset='OCHumanVal')
        test(model, dataset='OCHumanTest') 
