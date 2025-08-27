# --------------------------------------------------------------------------------------------
# Modified from CAFE codebase (https://github.com/dk-kim/CAFE_codebase)
# Copyright (c) Kim, Dongkeun; Song, Youngkil; Cho, Minsu; Kwak, Suha . All Rights Reserved
# --------------------------------------------------------------------------------------------
from collections import defaultdict
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


# Define activity classes
ACTIONS = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking']  # not 0 indexed
ACTIVITIES = ['NA', 'Crossing', 'Waiting', 'Queueing', 'Walking', 'Talking']

# Frame counts for each sequence
FRAMES_NUM = {1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302,
              11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342,
              21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356,
              31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401,
              41: 707, 42: 420, 43: 410, 44: 356}

FRAMES_SIZE = {1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720),
               8: (480, 720), 9: (480, 720), 10: (480, 720),
               11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720),
               17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800),
               21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720),
               27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720),
               31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720),
               37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720),
               41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}

def collective_read_annotations(path, sequences, num_class, image_width=None, image_height=None):
    """
    Read annotations from Collective dataset

    Args:
        path: Path to the CAD dataset root directory
        sequences: List of sequence names (e.g., [1, 2, 3, ...])
        num_class: Number of individual action classes
        image_width: Target image width for scaling
        image_height: Target image height for scaling

    Returns:
        labels: Dictionary containing annotations for each sequence
    """
    labels = {}
    annotations_path = os.path.join(path, 'annotations')

    for seq in sequences:
        seq_num = seq  # Already integer
        annotation_file = os.path.join(annotations_path, f'{seq}_annotations.txt')

        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file {annotation_file} not found")
            continue

        # Read annotation file
        frame_data = defaultdict(list)

        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 9:
                    continue

                frame_id = int(parts[0])
                x1, y1, x2, y2 = map(float, parts[1:5])
                individual_action = int(parts[5])
                social_activity = int(parts[6])
                track_id = int(parts[7])
                group_id = int(parts[8])

                frame_data[frame_id].append({
                    'bbox': [x1, y1, x2, y2],
                    'individual_action': individual_action,
                    'social_activity': social_activity,
                    'track_id': track_id,
                    'group_id': group_id
                })
        for frame_id, actors in frame_data.items():
            # Get original image dimensions for this sequence
            original_height, original_width = FRAMES_SIZE[seq_num]

            image_width = image_width if image_width is not None else original_width
            image_height = image_height if image_height is not None else original_height

            # Calculate scaling factors
            width_scale = image_width / original_width
            height_scale = image_height / original_height

            boxes, actions, membership, activities, members = [], [], [], [], []
            groups = {}

            # Sort actors by track_id for consistency
            actors.sort(key=lambda x: x['track_id'])

            # First pass: count group sizes
            group_counts = {}
            for actor in actors:
                group_id = actor['group_id']
                group_counts[group_id] = group_counts.get(group_id, 0) + 1

            # Create mapping for singleton groups to negative IDs
            singleton_counter = -2
            group_id_mapping = {}
            for group_id, count in group_counts.items():
                if count == 1:  # Singleton group
                    group_id_mapping[group_id] = singleton_counter
                    singleton_counter -= 1
                else:
                    group_id_mapping[group_id] = group_id

            for i, actor in enumerate(actors):
                x1, y1, x2, y2 = actor['bbox']

                # Scale coordinates to match target image dimensions
                x1 = x1 * width_scale
                y1 = y1 * height_scale
                x2 = x2 * width_scale
                y2 = y2 * height_scale

                # Convert to center coordinates and width/height
                x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                boxes.append([x_c, y_c, w, h])

                # Individual action
                individual_action = actor['individual_action'] - 1  # Convert to 0-indexed
                if individual_action >= len(ACTIONS):
                    raise ValueError(
                        f"Individual action {individual_action} exceeds defined actions length {len(ACTIONS)}")
                actions.append(individual_action)

                # Group membership - use mapped group ID
                original_group_id = actor['group_id']
                mapped_group_id = group_id_mapping[original_group_id]
                membership.append(mapped_group_id)

                # Social activity
                social_activity = actor['social_activity']
                if social_activity >= len(ACTIVITIES):
                    raise ValueError(
                        f"Social activity {social_activity} exceeds defined activities length {len(ACTIVITIES)}")

                # Track groups using mapped group ID
                if mapped_group_id not in groups:
                    groups[mapped_group_id] = {
                        'activity': social_activity,
                        'members': torch.zeros(len(actors))
                    }
                groups[mapped_group_id]['members'][i] = 1

            # Extract group activities and members
            for group_id in sorted(groups.keys()):
                activities.append(groups[group_id]['activity'])
                members.append(groups[group_id]['members'])

            # Convert to tensors
            actions = torch.tensor(actions, dtype=torch.long)
            boxes = torch.tensor(boxes, dtype=torch.float)
            membership = torch.tensor(membership, dtype=torch.long) - 1  # 0-indexed

            # # Remap membership IDs to frame unique.
            # _, membership_compact = torch.unique(membership, return_inverse=True)
            # membership = membership_compact

            activities = torch.tensor(activities, dtype=torch.long)

            if len(members) == 0:
                members = torch.tensor([])
            else:
                members = torch.stack(members).float()

            annotations = {
                'boxes': boxes,
                'actions': actions,
                'membership': membership,
                'activities': activities,
                'members': members,
                'num_frames': FRAMES_NUM[seq_num],
                'interval': 1,  # Assuming consecutive frames
                'key_frame': frame_id,
            }

            labels[(seq_num, frame_id)] = annotations

    return labels
        # Process each frame
    #     for frame_id, actors in frame_data.items():
    #         # Get original image dimensions for this sequence
    #         original_height, original_width = FRAMES_SIZE[seq_num]
    #
    #         image_width = image_width if image_width is not None else original_width
    #         image_height = image_height if image_height is not None else original_height
    #
    #         # Calculate scaling factors
    #         width_scale = image_width / original_width
    #         height_scale = image_height / original_height
    #
    #         boxes, actions, membership, activities, members = [], [], [], [], []
    #         groups = {}
    #
    #         # Sort actors by track_id for consistency
    #         actors.sort(key=lambda x: x['track_id'])
    #
    #         for i, actor in enumerate(actors):
    #             x1, y1, x2, y2 = actor['bbox']
    #
    #             # Scale coordinates to match target image dimensions
    #             x1 = x1 * width_scale
    #             y1 = y1 * height_scale
    #             x2 = x2 * width_scale
    #             y2 = y2 * height_scale
    #
    #             # Convert to center coordinates and width/height
    #             x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
    #             w, h = x2 - x1, y2 - y1
    #             boxes.append([x_c, y_c, w, h])
    #
    #             # Individual action
    #             individual_action = actor['individual_action']-1  # Convert to 0-indexed
    #             if individual_action >= len(ACTIONS):
    #                 raise ValueError(
    #                     f"Individual action {individual_action} exceeds defined actions length {len(ACTIONS)}")
    #             actions.append(individual_action)
    #
    #             # Group membership
    #             group_id = actor['group_id']
    #             membership.append(group_id)
    #
    #             # Social activity
    #             social_activity = actor['social_activity']
    #             if social_activity >= len(ACTIVITIES):
    #                 raise ValueError(
    #                     f"Social activity {social_activity} exceeds defined activities length {len(ACTIVITIES)}")
    #
    #             # Track groups
    #             if group_id not in groups:
    #                 groups[group_id] = {
    #                     'activity': social_activity,
    #                     'members': torch.zeros(len(actors))
    #                 }
    #             groups[group_id]['members'][i] = 1
    #
    #         # Extract group activities and members
    #         for group_id in sorted(groups.keys()):
    #             activities.append(groups[group_id]['activity'])
    #             members.append(groups[group_id]['members'])
    #
    #         # Convert to tensors
    #         actions = torch.tensor(actions, dtype=torch.long)
    #         boxes = torch.tensor(boxes, dtype=torch.float)
    #         membership = torch.tensor(membership, dtype=torch.long) - 1  # 0-indexed
    #
    #         # # Remap membership IDs to frame unique.
    #         # _, membership_compact = torch.unique(membership, return_inverse=True)
    #         # membership = membership_compact
    #
    #         activities = torch.tensor(activities, dtype=torch.long)
    #
    #         if len(members) == 0:
    #             members = torch.tensor([])
    #         else:
    #             members = torch.stack(members).float()
    #
    #         annotations = {
    #             'boxes': boxes,
    #             'actions': actions,
    #             'membership': membership,
    #             'activities': activities,
    #             'members': members,
    #             'num_frames': FRAMES_NUM[seq_num],
    #             'interval': 1,  # Assuming consecutive frames
    #             'key_frame': frame_id,
    #         }
    #
    #         labels[(seq_num, frame_id)] = annotations
    #
    # return labels


def collective_all_frames(labels):
    """Get all frame identifiers from labels"""
    frames = []
    for seq_id, anns in labels.items():
        frames.append(seq_id)
    return frames


class CollectiveDataset(data.Dataset):
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(CollectiveDataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.num_boxes = args.num_boxes
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_class = args.num_class
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        if self.num_frame == 1:
            frames = self.select_key_frames(self.frames[idx])
        else:
            frames = self.select_frames(self.frames[idx])

        samples = self.load_samples(frames)
        return samples

    def __len__(self):
        return len(self.frames)

    def select_key_frames(self, frame):
        """Select key frame for single frame processing"""
        annotation = self.anns[frame]
        key_frame = annotation['key_frame']
        return [(frame, int(key_frame))]

    def select_frames(self, frame):
        """Select multiple frames for temporal processing"""
        seq_num, frame_id = frame
        annotation = self.anns[frame]
        key_frame = annotation['key_frame']
        total_frames = FRAMES_NUM[seq_num]
        interval = annotation['interval']

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(1, total_frames + 1), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = total_frames // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
                sample_frames = [max(1, min(total_frames, f + 1)) for f in sample_frames]  # Ensure valid frame range
        else:
            if self.random_sampling:
                sample_frames = random.sample(range(1, total_frames + 1), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = total_frames // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
                sample_frames = [max(1, min(total_frames, f + 1)) for f in sample_frames]  # Ensure valid frame range

        return [((seq_num, frame_id), fid) for fid in sample_frames]

    def load_samples(self, frames):
        """Load image samples and corresponding annotations"""
        images, boxes, gt_boxes, actions, activities, members, membership = [], [], [], [], [], [], []
        targets = {}
        fids = []

        for i, (frame, fid) in enumerate(frames):
            seq_num, frame_id = frame
            fids.append(fid)

            # Load image
            img_path = os.path.join(self.image_path, f'seq{seq_num:02d}', f'frame{fid:04d}.jpg')

            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found")
                # Get original dimensions for this sequence
                original_height, original_width = FRAMES_SIZE[seq_num]
                img = Image.new('RGB', (original_width, original_height), color='black')
            else:
                img = Image.open(img_path)

            image_w, image_h = img.width, img.height
            img = self.transform(img)
            images.append(img)

            # Get annotations for this frame
            num_boxes = self.anns[frame]['boxes'].shape[0]

            # Ground truth boxes (normalized)
            for box in self.anns[frame]['boxes']:
                x_c, y_c, w, h = box
                gt_boxes.append([x_c / image_w, y_c / image_h, w / image_w, h / image_h])

            # For collective dataset, we use the same boxes as ground truth
            # since we don't have separate tracking data
            temp_boxes = self.anns[frame]['boxes'].clone().detach()

            # Normalize boxes
            temp_boxes[:, 0] /= image_w  # x_c
            temp_boxes[:, 1] /= image_h  # y_c
            temp_boxes[:, 2] /= image_w  # w
            temp_boxes[:, 3] /= image_h  # h

            boxes.append(temp_boxes.numpy())
            actions.append(self.anns[frame]['actions'])
            activities.append(self.anns[frame]['activities'])
            members.append(self.anns[frame]['members'])
            membership.append(self.anns[frame]['membership'])

            # Pad to match num_boxes (assuming current boxes <= num_boxes)
            if len(boxes[-1]) != self.num_boxes:
                padding = np.zeros((self.num_boxes - len(boxes[-1]), 4))
                boxes[-1] = np.vstack([boxes[-1], padding])

            if len(actions[-1]) != self.num_boxes:
                # Pad with 0 as dummy actions
                padding = torch.tensor([0] * (self.num_boxes - len(actions[-1])))
                actions[-1] = torch.cat((actions[-1], padding))

            if members[-1].numel() > 0 and members[-1].shape[1] != self.num_boxes:
                # Pad with zeros
                padding = torch.zeros((members[-1].shape[0], self.num_boxes - members[-1].shape[1]))
                members[-1] = torch.hstack((members[-1], padding))

            if len(membership[-1]) != self.num_boxes:
                # Pad with -1 (no group)
                padding = torch.tensor([-1] * (self.num_boxes - len(membership[-1])))
                membership[-1] = torch.cat((membership[-1], padding))

        # Stack all data
        images = torch.stack(images)
        boxes = np.stack(boxes).reshape([self.num_frame, -1, 4])
        gt_boxes = np.stack(gt_boxes).reshape([self.num_frame, -1, 4])
        actions = torch.stack(actions)
        membership = torch.stack(membership)

        if len(activities) == 0:
            activities = torch.tensor([])
            members = torch.tensor([])
        else:
            activities = torch.stack(activities)
            if members[0].numel() > 0:
                members = torch.stack(members)
            else:
                members = torch.tensor([])

        boxes = torch.from_numpy(boxes).float()
        gt_boxes = torch.from_numpy(gt_boxes).float()

        targets = {
            'actions': actions,
            'activities': activities,
            'boxes': boxes,
            'gt_boxes': gt_boxes,
            'members': members,
            'membership': membership
        }

        infos = {
            'vid': seq_num,
            'sid': frame_id,
            'fid': fids,
            'key_frame': self.anns[frame]['key_frame']
        }

        return images, targets, infos