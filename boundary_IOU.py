# -*- coding = utf-8 -*-
# @File Name : boundary IOU
# @Date : 2024/10/31 07:30
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from sklearn.metrics import precision_recall_curve, auc


def extract_3d_boundary(volume, dilation_iterations=1):
    """
    Extracts the boundary of a 3D binary volume using binary dilation.
    Parameters:
        volume (np.ndarray): 3D binary volume (segmentation mask).
        dilation_iterations (int): Number of dilation iterations to create a boundary band.
    Returns:
        np.ndarray: 3D binary volume with boundary voxels highlighted.
    """
    # Dilate the volume to create a band around the boundary
    dilated_volume = binary_dilation(volume, iterations=dilation_iterations)
    # Boundary is the difference between the dilated volume and the original volume
    boundary = dilated_volume & ~volume
    return boundary


def boundary_iou(prediction, ground_truth, dilation_iterations=1):
    """
    Calculates the Boundary IoU between the predicted and ground truth 3D volumes.
    Parameters:
        prediction (np.ndarray): Predicted 3D binary volume.
        ground_truth (np.ndarray): Ground truth 3D binary volume.
        dilation_iterations (int): Number of dilation iterations to create a boundary band.
    Returns:
        float: Boundary IoU score.
    """
    # Extract boundaries for prediction and ground truth volumes
    boundary_pred = extract_3d_boundary(prediction, dilation_iterations)
    boundary_gt = extract_3d_boundary(ground_truth, dilation_iterations)

    # Calculate Intersection and Union for boundary regions
    intersection = np.logical_and(boundary_pred, boundary_gt).sum()
    union = np.logical_or(boundary_pred, boundary_gt).sum()

    # Boundary IoU score
    boundary_iou_score = intersection / union if union != 0 else 0
    return boundary_iou_score


def boundary_average_precision(prediction, confidence_scores, ground_truth, dilation_iterations=1):
    """
    Calculates the Boundary Average Precision (AP) between the predicted and ground truth 3D volumes.
    Parameters:
        prediction (np.ndarray): Predicted 3D binary volume.
        confidence_scores (np.ndarray): Confidence scores for each voxel in the prediction.
        ground_truth (np.ndarray): Ground truth 3D binary volume.
        dilation_iterations (int): Number of dilation iterations to create a boundary band.
    Returns:
        float: Boundary Average Precision (AP).
    """
    # Extract boundaries
    boundary_pred = extract_3d_boundary(prediction, dilation_iterations)
    boundary_gt = extract_3d_boundary(ground_truth, dilation_iterations)

    # Flatten boundaries and confidence scores for calculating AP
    boundary_pred_flat = boundary_pred.flatten()
    boundary_gt_flat = boundary_gt.flatten()
    confidence_scores_flat = confidence_scores.flatten()

    # Only consider boundary voxels for AP calculation
    boundary_conf_scores = confidence_scores_flat[boundary_pred_flat == 1]
    boundary_ground_truth = boundary_gt_flat[boundary_pred_flat == 1]

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(boundary_ground_truth, boundary_conf_scores)

    # Calculate Average Precision (AP) as the area under the precision-recall curve
    boundary_ap_score = auc(recall, precision)

    return boundary_ap_score


def boundary_pq(prediction, ground_truth, dilation_iterations=1):
    """
    Calculates the Boundary Panoptic Quality (Boundary PQ) between predicted and ground truth 3D volumes.
    Parameters:
        prediction (np.ndarray): Predicted 3D binary volume.
        ground_truth (np.ndarray): Ground truth 3D binary volume.
        dilation_iterations (int): Number of dilation iterations to create a boundary band.
    Returns:
        float: Boundary PQ score.
    """
    # Extract boundaries
    boundary_pred = extract_3d_boundary(prediction, dilation_iterations)
    boundary_gt = extract_3d_boundary(ground_truth, dilation_iterations)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) in boundary regions
    tp = np.logical_and(boundary_pred, boundary_gt).sum()  # True Positives: Overlapping boundary voxels
    fp = np.logical_and(boundary_pred, ~boundary_gt).sum()  # False Positives: Extra predicted boundary voxels
    fn = np.logical_and(~boundary_pred, boundary_gt).sum()  # False Negatives: Missed boundary voxels

    # Calculate Intersection over Union (IoU) for boundaries
    iou_boundary = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0

    # Calculate Boundary PQ using the formula
    boundary_pq_score = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) != 0 else 0

    return boundary_pq_score


if __name__ == '__main__':
    # Example usage
    # Assume 'pred' and 'gt' are 3D binary numpy arrays for the predicted and ground truth segmentations
    pred = np.random.randint(0, 2, size=(64, 64, 64))  # Example prediction volume
    gt = np.random.randint(0, 2, size=(64, 64, 64))  # Example ground truth volume
    confidence_scores = np.random.rand(64, 64, 64)                  # Random confidence scores for demonstration

    # Calculate Boundary IoU
    boundary_iou_score = boundary_iou(pred, gt, dilation_iterations=2)
    print("Boundary IoU Score:", boundary_iou_score)

    # Calculate Boundary AP
    boundary_ap_score = boundary_average_precision(pred, confidence_scores, gt, dilation_iterations=2)
    print("Boundary Average Precision (AP) Score:", boundary_ap_score)
