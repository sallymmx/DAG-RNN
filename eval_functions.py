import numpy as np

def cal_results(gt_labels,predictions,threshold):
  probs = np.array(predictions)
  gt_labels = np.array(gt_labels)
  #probs = 1 / (1 + np.exp(-predictions))
  mAP_voc = AP_VOC(gt_labels,probs)
  P_C,R_C,F1_C = pre_rec_f1(gt_labels, probs>threshold)
  labels1 = np.reshape(gt_labels,(np.product(gt_labels.shape),1))
  probs1 = np.reshape(probs>threshold,(np.product(probs.shape),1))
  P_O,R_O,F1_O = pre_rec_f1(labels1,probs1)
  out = {'mAP_voc':mAP_voc,'P_C':P_C,'R_C': R_C, 'F1_C':F1_C,'P_O':P_O,'R_O':R_O,'F1_O':F1_O}
  return out


def AP_VOC(labels,probs):
  #calc AP for each column of inputs
  num_cls = labels.shape[1]
  AP_voc = np.zeros((num_cls, 1));
  for m in range(num_cls):
     gt = labels[:,m]
     out = probs[:,m]
     #compute precision/recall
     si = np.argsort(out)
     si = si[::-1]
     tp = gt[si]
     fp = 1-gt[si]
     fp =np.cumsum(fp)
     tp = np.cumsum(tp)
     tp = tp.astype(float)
     fp = fp.astype(float)     
     rec = tp/sum(gt)
     prec = tp/(fp+tp)
     #compute voc12 style average precision
     ap = voc_ap(rec,prec);
     AP_voc[m] = ap 
  return AP_voc

def voc_ap(rec, prec):
  """
  ap = voc_ap(rec, prec)
  Computes the AP under the precision recall curve.
  """
  rec = rec.reshape(rec.size,1); prec = prec.reshape(prec.size,1)
  z = np.zeros((1,1)); o = np.ones((1,1));
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap

def pre_rec_f1(labels,preds):
  #calc P, R and F1 score for each column (class) of inputs
  #labels:         ground truth labels, type: logical, size: num_im*num_class
  #label_pred:     predicted labels, type: logical, size: num_im*num_class
  tp = np.logical_and(labels, preds)
  num_tp = np.sum(tp, 0) + np.finfo(float).eps
  num_pred = np.sum(preds, 0)+ np.finfo(float).eps
  num_p = np.sum(labels, 0)+ np.finfo(float).eps
  P_class = num_tp / num_pred;
  R_class = num_tp / num_p + np.finfo(float).eps
  F1_class = 2 * P_class * R_class / (P_class + R_class)  
  return  P_class,R_class,F1_class
