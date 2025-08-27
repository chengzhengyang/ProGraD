# ------------------------------------------------------------------------
# Modified from JRDB-ACT (https://github.com/JRDB-dataset/jrdb_toolkit/tree/main/Action%26Social_grouping_eval)
# ------------------------------------------------------------------------
from collections import Counter, defaultdict
import copy

import numpy as np


def make_image_key(v_id, c_id, f_id):
    """Returns a unique identifier for a video id & clip id & frame id"""
    return "%d,%d,%d" % (int(v_id), int(c_id), int(f_id))


def make_clip_key(image_key):
    """Returns a unique identifier for a video id & clip id"""
    v_id = image_key.split(',')[0]
    c_id = image_key.split(',')[1]
    return "%d,%d" % (int(v_id), int(c_id))


def read_text_file(text_file, eval_type, mode):
    """Loads boxes and class labels from a CSV file in the cafe format.

    Args:
      text_file: A file object.
      mode: 'gt' or 'pred'
      eval_type: 
        'gt_base': Eval type for trained model with ground turth actor tracklets as inputs.
        'detect_base': Eval type for trained model with tracker actor tracklets as inputs.

    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      g_labels: A dictionary mapping each unique image key (string) to a list of
        integer group id labels, matching the corresponding box in 'boxes'.
      act_labels: A dictionary mapping each unique image key (string) to a list of
        integer group activity class lables, matching the corresponding box in `boxes`.
      a_scores: A dictionary mapping each unique image key (string) to a list of
        actor confidence score values lables, matching the corresponding box in `boxes`.
      g_scores: A dictionary mapping each unique image key (string) to a list of
        group confidence score values lables, matching the corresponding box in `boxes`.
    """
    boxes = defaultdict(list)
    g_labels = defaultdict(list)
    act_labels = defaultdict(list)
    a_scores = defaultdict(list)
    g_scores = defaultdict(list)
    # reads each row in text file.
    with open(text_file.name) as r:
        for line in r.readlines():
            line = line.strip()
            if not line:  # If the line is empty, skip it.
                continue
            row = line.split()
            # makes image key.
            image_key = make_image_key(row[0], row[1], row[2])
            # box coordinates.
            x1, y1, x2, y2 = [float(n) for n in row[3:7]]
            # actor confidence score.
            if eval_type == 'detect_base' and mode == 'pred':
                a_score = float(row[10])
            else:
                a_score = 1.0
            # group confidence score.
            if mode == 'gt':
                g_score = None
            elif mode == 'pred':
                g_score = float(row[9])
            # group identity document.
            group_id = int(row[7])
            # group activity label.
            activity = int(row[8])

            boxes[image_key].append([x1, y1, x2, y2])
            g_labels[image_key].append(group_id)
            act_labels[image_key].append(activity)
            a_scores[image_key].append(a_score)
            g_scores[image_key].append(g_score)
    return boxes, g_labels, act_labels, a_scores, g_scores


def make_groups(boxes, g_labels, act_labels, g_scores):
    """combines boxes, activity, score to same group, same image

    Returns:
      groups_ids: A dictionary mapping each unique clip key (string) to a list of
        actor ids of each 'g_label'.
      groups_activity: A dictionary mapping each unique clip key (string) to a list of
        group activity class labels.
      groups_score: A dictionary mapping each unique clip key (string) to a list of
        group confidence score.
    """
    image_keys = boxes.keys()
    groups_activity = defaultdict(list)
    groups_score = defaultdict(list)
    groups_ids = defaultdict(list)
    frame_list = defaultdict(list)
    # makes clip key.
    for image_key in image_keys:
        clip_key = make_clip_key(image_key)
        frame_list[clip_key].append(image_key)
    clip_keys = frame_list.keys()
    for clip_key in clip_keys:
        group_ids = defaultdict(list)
        group_activity = defaultdict(set)
        group_score = defaultdict(set)
        for i, (g_label, act_label, g_score) in enumerate(
                zip(g_labels[frame_list[clip_key][0]], act_labels[frame_list[clip_key][0]],
                    g_scores[frame_list[clip_key][0]])):
            group_ids[g_label].append(i)
            group_activity[g_label].add(act_label)
            group_score[g_label].add(g_score)
        groups_ids[clip_key].append(group_ids)
        groups_activity[clip_key].append(group_activity)
        groups_score[clip_key].append(group_score)

    return groups_ids, groups_activity, groups_score


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
      labelmap_file: A file object containing a label map protocol buffer.

    Returns:
      labelmap: The label map in the form used by the object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
      class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    for line in labelmap_file:
        if line.startswith("  name:"):
            name = line.split('"')[1]
        elif line.startswith("  id:") or line.startswith("  label_id:"):
            class_id = int(line.strip().split(" ")[-1])
            labelmap.append({"id": class_id, "name": name})
            class_ids.add(class_id)
    return labelmap, class_ids


def IoU(box1, box2):
    """calculates IoU between two different boxes."""
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter + 1e-8)
    return iou


def cal_group_IoU(pred_group, gt_group):
    """calculates group IoU between two different groups"""
    # Intersection
    Intersection = sum([1 for det_id in pred_group[2] if det_id in gt_group[2]])

    # group IoU
    if Intersection != 0:
        group_IoU = Intersection / (len(pred_group[2]) + len(gt_group[2]) - Intersection)
    else:
        group_IoU = 0
    return group_IoU


def calculateAveragePrecision(rec, prec):
    """calculates AP score of each activity class by all-point interploation method."""
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ii = []

    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def outlier_metric(gt_groups_ids, gt_groups_activity, pred_groups_ids, pred_groups_activity, num_class):
    """calculates Outlier mIoU.

    Args:
      num_class: A number of activity classes.

    Returns:
      outlier_mIoU: Mean of outlier IoUs on each clip.
    """
    clip_IoU = defaultdict(list)
    TP = defaultdict(list)
    clip_keys = pred_groups_ids.keys()
    c_pred_groups_activity = copy.deepcopy(pred_groups_activity)
    c_gt_groups_activity = copy.deepcopy(gt_groups_activity)
    # prediction groups on each class. defines group has members equals or more than two.
    pred_groups = [[clip_key, group_id, pred_groups_ids[clip_key][0][group_id]] for clip_key in clip_keys if
                   clip_key in gt_groups_ids.keys() for group_id in pred_groups_ids[clip_key][0].keys() if
                   c_pred_groups_activity[clip_key][0][group_id].pop() == (num_class + 1)]
    # ground truth groups on each class.
    gt_groups = [[clip_key, group_id, gt_groups_ids[clip_key][0][group_id]] for clip_key in clip_keys if
                 clip_key in gt_groups_ids.keys() for group_id in gt_groups_ids[clip_key][0].keys() if
                 c_gt_groups_activity[clip_key][0][group_id].pop() == (num_class + 1)]
    for clip_key in clip_keys:
        # escapes error that there are not exist pred_image_key on gt.txt.
        if clip_key in gt_groups_ids.keys():
            # groups on same clip
            c_pred_groups = [pred_group for pred_group in pred_groups if pred_group[0] == clip_key]
            c_gt_groups = [gt_group for gt_group in gt_groups if gt_group[0] == clip_key]
            if len(c_pred_groups) != 0 and len(c_gt_groups) != 0:
                # outliers on prediction and ground truth.
                c_pred_ids = [pred_id for c_pred_group in c_pred_groups for pred_id in c_pred_group[2]]
                c_gt_ids = [gt_id for c_gt_group in c_gt_groups for gt_id in c_gt_group[2]]
                # number of True positive outliers.
                TP[clip_key] = sum([1 for pred_id in c_pred_ids if pred_id in c_gt_ids])
                clip_IoU[clip_key] = TP[clip_key] / (len(c_pred_ids) + len(c_gt_ids) - TP[clip_key])
                clip_IoU['total'].append(clip_IoU[clip_key])
            elif len(c_pred_groups) != 0 or len(c_gt_groups) != 0:
                TP[clip_key] = 0
                clip_IoU[clip_key] = 0
                clip_IoU['total'].append(clip_IoU[clip_key])
    # outlier mIoU.
    outlier_mIoU = np.array(clip_IoU['total']).mean()
    return outlier_mIoU * 100.0


def group_mAP_eval(gt_groups_ids, gt_groups_activity, pred_groups_ids, pred_groups_activity, pred_groups_scores,
                   categories, thresh):
    """calculates group mAP.

    Args:
      categories: A list of group activity classes, given as {name: ,id: }.
      thresh: A group IoU threshold for true positive prediction group condition.

    Returns:
      group_mAP: Mean of group APs on each activity class.
      group_APs; A list of each group AP on each activity class.
    """
    clip_keys = pred_groups_ids.keys()
    # acc on each class.
    group_APs = np.zeros(len(categories))
    for c, clas in enumerate(categories):
        # copy for set funtion to pop.
        c_pred_groups_activity = copy.deepcopy(pred_groups_activity)
        c_gt_groups_activity = copy.deepcopy(gt_groups_activity)
        # prediction groups on each class.
        pred_groups = [
            [clip_key, group_id, pred_groups_ids[clip_key][0][group_id], pred_groups_scores[clip_key][0][group_id]] for
            clip_key in clip_keys if clip_key in gt_groups_ids.keys() for group_id in
            pred_groups_ids[clip_key][0].keys() if
                                     c_pred_groups_activity[clip_key][0][group_id].pop() == clas['id'] and len(
                                         pred_groups_ids[clip_key][0][group_id]) >= 2]
        # ground truth groups on each class.
        gt_groups = [[clip_key, group_id, gt_groups_ids[clip_key][0][group_id]] for clip_key in clip_keys if
                     clip_key in gt_groups_ids.keys() for group_id in gt_groups_ids[clip_key][0].keys() if
                     c_gt_groups_activity[clip_key][0][group_id].pop() == clas['id'] and len(
                         gt_groups_ids[clip_key][0][group_id]) >= 2]

        # denominator of Recall.
        npos = len(gt_groups)

        # sorts det_groups in descending order for g_score.
        pred_groups = sorted(pred_groups, key=lambda conf: conf[3], reverse=True)

        TP = np.zeros(len(pred_groups))
        FP = np.zeros(len(pred_groups))

        det = Counter(gt_group[0] for gt_group in gt_groups)

        for key, val in det.items():
            det[key] = np.zeros(val)

        # AP matching algorithm.
        for p, pred_group in enumerate(pred_groups):
            if pred_group[0] in gt_groups_ids.keys():
                gt = [gt_group for gt_group in gt_groups if gt_group[0] == pred_group[0]]
                group_IoU_Max = 0
                for j, gt_group in enumerate(gt):
                    group_IoU = cal_group_IoU(pred_group, gt_group)
                    if group_IoU > group_IoU_Max:
                        group_IoU_Max = group_IoU
                        jmax = j
                # true positive prediction group condition.
                if group_IoU_Max >= thresh:
                    if det[pred_group[0]][jmax] == 0:
                        TP[p] = 1
                        det[pred_group[0]][jmax] = 1
                    else:
                        FP[p] = 1
                else:
                    FP[p] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        # recall
        rec = acc_TP / npos
        # precision
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        # group AP on each group activity class
        group_APs[c] = ap * 100
        # group mAP
    group_mAP = group_APs.mean()
    return group_mAP, group_APs


class GAD_Evaluation():
    def __init__(self, args):
        super(GAD_Evaluation, self).__init__()
        self.eval_type = args.eval_type
        self.categories, self.class_whitelist = read_labelmap(args.labelmap)
        self.gt_boxes, self.gt_g_labels, self.gt_act_labels, _, self.gt_g_scores = read_text_file(args.groundtruth,
                                                                                                  self.eval_type,
                                                                                                  mode='gt')
        self.gt_groups_ids, self.gt_groups_activity, _ = make_groups(
            self.gt_boxes, self.gt_g_labels, self.gt_act_labels, self.gt_g_scores)

    def evaluate(self, detections):
        pred_boxes, pred_g_labels, pred_act_labels, pred_a_scores, pred_g_scores = read_text_file(detections,
                                                                                                  self.eval_type,
                                                                                                  mode='pred')
        pred_groups_ids, pred_groups_activity, pred_groups_scores = make_groups(pred_boxes, pred_g_labels,
                                                                                pred_act_labels,
                                                                                pred_g_scores)
        group_mAP, group_APs = group_mAP_eval(self.gt_groups_ids, self.gt_groups_activity,
                                              pred_groups_ids, pred_groups_activity, pred_groups_scores,
                                              self.categories, thresh=1.0)
        group_mAP_2, group_APs_2 = group_mAP_eval(self.gt_groups_ids, self.gt_groups_activity,
                                                  pred_groups_ids, pred_groups_activity, pred_groups_scores,
                                                  self.categories, thresh=0.5)
        outlier_mIoU = outlier_metric(self.gt_groups_ids, self.gt_groups_activity,
                                      pred_groups_ids, pred_groups_activity,
                                      len(self.categories))
        result = {
            'group_APs_1.0': group_APs,
            'group_mAP_1.0': group_mAP,
            'group_APs_0.5': group_APs_2,
            'group_mAP_0.5': group_mAP_2,
            'outlier_mIoU': outlier_mIoU,
        }
        return result


def individual_action_accuracy(gt_boxes, gt_act_labels, pred_boxes, pred_act_labels):
    """
    Calculate Individual Action Accuracy with direct GT-Pred correspondence.

    Args:
        gt_boxes: Ground truth boxes from GAD's read_text_file
        gt_act_labels: Ground truth action labels from GAD's read_text_file
        pred_boxes: Predicted boxes from GAD's read_text_file (same as GT boxes)
        pred_act_labels: Predicted action labels from GAD's read_text_file

    Returns:
        accuracy: Individual action accuracy (0-100)
        correct: Number of correct predictions
        total: Total number of predictions
    """
    correct = total = 0

    for frame_key, gt_box_list in gt_boxes.items():
        if frame_key not in pred_boxes or frame_key not in pred_act_labels:
            continue

        # Sort by x1 coordinate elegantly
        gt_sorted = sorted(zip(gt_box_list, gt_act_labels[frame_key]), key=lambda x: x[0][0])
        pred_sorted = sorted(zip(pred_boxes[frame_key], pred_act_labels[frame_key]), key=lambda x: x[0][0])

        # Compare actions directly
        min_len = min(len(gt_sorted), len(pred_sorted))
        for i in range(min_len):
            total += 1
            if gt_sorted[i][1] == pred_sorted[i][1]:  # compare action labels
                correct += 1

    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return accuracy, correct, total


def membership_accuracy(gt_groups_ids, pred_groups_ids, gt_boxes, pred_boxes):
    """
    Calculate Membership Accuracy with direct GT-Pred correspondence.
    Args:
        gt_groups_ids: Ground truth group IDs from GAD's make_groups
        pred_groups_ids: Predicted group IDs from GAD's make_groups
    Returns:
        accuracy: Membership accuracy (0-100)
        correct: Number of correct predictions
        total: Total number of predictions
    """
    correct = 0
    total = 0

    for clip_key in gt_groups_ids.keys():
        if clip_key not in pred_groups_ids:
            continue
        # print(gt_groups_ids.keys())
        gt_groups = gt_groups_ids[clip_key][0]
        pred_groups = pred_groups_ids[clip_key][0]

        x, y = clip_key.split(',')
        box_key = f"{x},{y},{y}"
        gt_box_list = gt_boxes[box_key]
        pred_box_list = pred_boxes[box_key]

        # Create position-to-group mappings
        def create_position_to_group_map(groups, box_list):
            # Collect all actors and their groups
            actor_to_group = {}
            for group_id, actor_list in groups.items():
                for actor_id in actor_list:
                    actor_to_group[actor_id] = group_id

            # Sort by x1 coordinate and create position mapping
            sorted_positions = sorted(range(len(box_list)), key=lambda i: box_list[i][0])
            pos_to_group = {}
            for pos_idx, box_idx in enumerate(sorted_positions):
                if box_idx in actor_to_group:
                    pos_to_group[pos_idx] = actor_to_group[box_idx]

            return pos_to_group

        gt_pos_to_group = create_position_to_group_map(gt_groups, gt_box_list)
        pred_pos_to_group = create_position_to_group_map(pred_groups, pred_box_list)

        # Create group-to-positions mappings
        def pos_to_group_sets(pos_to_group):
            group_to_pos = defaultdict(set)
            for pos, group_id in pos_to_group.items():
                group_to_pos[group_id].add(pos)
            return set(frozenset(positions) for positions in group_to_pos.values())

        gt_group_sets = pos_to_group_sets(gt_pos_to_group)
        pred_group_sets = pos_to_group_sets(pred_pos_to_group)

        # All positions that have groups assigned
        all_positions = set(gt_pos_to_group.keys()) | set(pred_pos_to_group.keys())
        total += len(all_positions)

        # Find intersection of matching groups
        matched_groups = gt_group_sets & pred_group_sets
        correct += sum(len(group) for group in matched_groups)

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total

def group_accuracy(gt_groups_ids, pred_groups_ids, gt_boxes, pred_boxes,
                   gt_groups_activity=None, pred_groups_activity=None):
    """
    Calculate Group Accuracy with direct GT-Pred correspondence.
    Args:
        gt_groups_ids: Ground truth group IDs from GAD's make_groups
        pred_groups_ids: Predicted group IDs from GAD's make_groups
        gt_boxes: Ground truth bounding boxes
        pred_boxes: Predicted bounding boxes
        gt_groups_activity: Ground truth group activities (optional, for social accuracy)
        pred_groups_activity: Predicted group activities (optional, for social accuracy)
    Returns:
        dict with:
            - membership_accuracy: Membership accuracy (0-100)
            - social_accuracy: Social accuracy (0-100) if activities provided
            - membership_correct/total: counts for membership
            - social_correct/total: counts for social if activities provided
    """
    from collections import defaultdict

    membership_correct = 0
    social_correct = 0
    total = 0

    # Check if we're calculating social accuracy
    calculate_social = (gt_groups_activity is not None and pred_groups_activity is not None)

    for clip_key in gt_groups_ids.keys():
        if clip_key not in pred_groups_ids:
            continue

        gt_groups = gt_groups_ids[clip_key][0]
        pred_groups = pred_groups_ids[clip_key][0]

        x, y = clip_key.split(',')
        box_key = f"{x},{y},{y}"
        gt_box_list = gt_boxes[box_key]
        pred_box_list = pred_boxes[box_key]

        # Create position-to-group mappings
        def create_position_to_group_map(groups, box_list):
            actor_to_group = {}
            for group_id, actor_list in groups.items():
                for actor_id in actor_list:
                    actor_to_group[actor_id] = group_id

            sorted_positions = sorted(range(len(box_list)), key=lambda i: box_list[i][0])
            pos_to_group = {}
            for pos_idx, box_idx in enumerate(sorted_positions):
                if box_idx in actor_to_group:
                    pos_to_group[pos_idx] = actor_to_group[box_idx]

            return pos_to_group

        gt_pos_to_group = create_position_to_group_map(gt_groups, gt_box_list)
        pred_pos_to_group = create_position_to_group_map(pred_groups, pred_box_list)

        # Separate positions based on GT group ID
        normal_positions = {pos: group_id for pos, group_id in gt_pos_to_group.items() if group_id > -1}
        singleton_positions = {pos: group_id for pos, group_id in gt_pos_to_group.items() if group_id <= -1}

        # Convert to sets of position groups with activities for normal positions
        def pos_to_group_sets(pos_to_group):
            group_to_pos = defaultdict(set)
            for pos, group_id in pos_to_group.items():
                group_to_pos[group_id].add(pos)
            return set(frozenset(positions) for positions in group_to_pos.values())

        # Get group sets for membership (only normal positions)
        gt_group_sets = pos_to_group_sets(normal_positions)
        pred_group_sets_normal = pos_to_group_sets(
            {pos: group_id for pos, group_id in pred_pos_to_group.items() if pos in normal_positions})

        # All positions that have groups assigned
        all_positions = set(gt_pos_to_group.keys()) | set(pred_pos_to_group.keys())
        total += len(all_positions)

        # Find intersection of matching groups (membership only) for normal positions
        matched_groups = gt_group_sets & pred_group_sets_normal
        membership_correct += sum(len(group) for group in matched_groups)

        # Handle singleton positions (GT group ID <= -1) - they are always considered correct for membership
        membership_correct += len(singleton_positions)

        # Calculate social accuracy if activities provided
        if calculate_social:
            # Get group sets with activities
            gt_activity_dict = gt_groups_activity[clip_key][0] if clip_key in gt_groups_activity else {}
            pred_activity_dict = pred_groups_activity[clip_key][0] if clip_key in pred_groups_activity else {}

            # Create a modified function that uses the activity dict directly
            def pos_to_group_sets_with_activity(pos_to_group, activity_dict):
                group_to_pos = defaultdict(set)
                for pos, group_id in pos_to_group.items():
                    group_to_pos[group_id].add(pos)

                result = set()
                for group_id, positions in group_to_pos.items():
                    activity = activity_dict.get(group_id, '')
                    frozen_positions = frozenset(positions)

                    # Convert activity to hashable type
                    if isinstance(activity, set):
                        frozen_activity = frozenset(activity)
                    elif isinstance(activity, (list, dict)):
                        frozen_activity = frozenset(activity) if isinstance(activity, list) else frozenset(
                            activity.items())
                    else:
                        frozen_activity = activity  # Already hashable

                    result.add((frozen_positions, frozen_activity))
                return result

            # For normal positions, check both membership and activity
            gt_group_activity_sets_normal = pos_to_group_sets_with_activity(normal_positions, gt_activity_dict)
            pred_group_activity_sets_normal = pos_to_group_sets_with_activity(
                {pos: group_id for pos, group_id in pred_pos_to_group.items() if pos in normal_positions},
                pred_activity_dict
            )

            # Find intersection of matching groups with activities for normal positions
            matched_groups_with_activity = gt_group_activity_sets_normal & pred_group_activity_sets_normal
            social_correct += sum(len(group_positions) for group_positions, activity in matched_groups_with_activity)

            # For singleton positions (GT group ID <= -1), only check activity labels
            for pos in singleton_positions:
                if pos in pred_pos_to_group:
                    pred_group_id = pred_pos_to_group[pos]
                    gt_group_id = gt_pos_to_group[pos]

                    # Get activities for comparison
                    gt_activity = gt_activity_dict.get(gt_group_id, '')
                    pred_activity = pred_activity_dict.get(pred_group_id, '')

                    # Convert activities to comparable format
                    def normalize_activity(activity):
                        if isinstance(activity, set):
                            return frozenset(activity)
                        elif isinstance(activity, list):
                            return frozenset(activity)
                        elif isinstance(activity, dict):
                            return frozenset(activity.items())
                        else:
                            return activity

                    if normalize_activity(gt_activity) == normalize_activity(pred_activity):
                        social_correct += 1

    # Calculate accuracies
    membership_accuracy = (membership_correct / total * 100) if total > 0 else 0

    result = {
        'membership_accuracy': membership_accuracy,
        'membership_correct': membership_correct,
        'membership_total': total
    }

    if calculate_social:
        social_accuracy = (social_correct / total * 100) if total > 0 else 0
        result.update({
            'social_accuracy': social_accuracy,
            'social_correct': social_correct,
            'social_total': total
        })

    return result

def membership_accuracy(gt_groups_ids, pred_groups_ids, gt_boxes, pred_boxes):
    """Calculate only membership accuracy"""
    result = group_accuracy(gt_groups_ids, pred_groups_ids, gt_boxes, pred_boxes)
    return result['membership_accuracy'], result['membership_correct'], result['membership_total']


def membership_social_accuracy(gt_groups_ids, pred_groups_ids, gt_boxes, pred_boxes,
                    gt_groups_activity, pred_groups_activity):
    """Calculate both membership and social accuracy"""
    result = group_accuracy(gt_groups_ids, pred_groups_ids, gt_boxes, pred_boxes,
                            gt_groups_activity, pred_groups_activity)
    return (result['membership_accuracy'], result['membership_correct'], result['membership_total'],
            result['social_accuracy'], result['social_correct'], result['social_total'])


class SocialCAD_Evaluation():
    def __init__(self, args):
        super(SocialCAD_Evaluation, self).__init__()
        self.eval_type = args.eval_type

        # Load GT data
        self.gt_boxes, self.gt_g_labels, self.gt_act_labels, _, self.gt_g_scores = read_text_file(
            args.groundtruth, self.eval_type, mode='gt'
        )

        # Build GT groups
        self.gt_groups_ids, self.gt_groups_activity, _ = make_groups(
            self.gt_boxes, self.gt_g_labels, self.gt_act_labels, self.gt_g_scores
        )

    def evaluate(self, predictions_file):
        # Load predictions
        pred_boxes, pred_g_labels, pred_act_labels, pred_a_scores, pred_g_scores = read_text_file(
            predictions_file, self.eval_type, mode='pred'
        )

        # Build predicted groups
        pred_groups_ids, pred_groups_activity, pred_groups_scores = make_groups(
            pred_boxes, pred_g_labels, pred_act_labels, pred_g_scores
        )

        # Calculate custom metrics (no actor matching needed)
        ind_acc, ind_correct, ind_total = individual_action_accuracy(
            self.gt_boxes, self.gt_act_labels,
            pred_boxes, pred_act_labels)

        # mem_acc, mem_correct, mem_total = membership_accuracy(
        #     self.gt_groups_ids, pred_groups_ids, self.gt_boxes, pred_boxes)

        (mem_acc, mem_correct, mem_total, soc_acc, soc_correct, soc_total) = membership_social_accuracy(
            self.gt_groups_ids, pred_groups_ids, self.gt_boxes,
            pred_boxes, self.gt_groups_activity, pred_groups_activity)

        # Return results in GAD format
        result = {
            'Individual_Action_Accuracy': ind_acc,
            'Individual_Action_Correct': ind_correct,
            'Individual_Action_Total': ind_total,
            'Membership_Accuracy': mem_acc,
            'Membership_Correct': mem_correct,
            'Membership_Total': mem_total,
            'Social_Accuracy': soc_acc,
            'Social_Correct': soc_correct,
            'Social_Total': soc_total,
        }

        return result
