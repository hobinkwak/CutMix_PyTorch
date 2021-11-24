import numpy as np
import scipy.stats as stats


def cutmix(self, images, labels):
    lmbda = stats.beta(1, 1).rvs(1)[0]
    H = images[0].size()[-2]
    W = images[0].size()[-1]
    cut_rat = np.sqrt(1 - lmbda)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    center_x = np.random.randint(W)
    center_y = np.random.randint(H)
    boundary_x1 = np.clip(center_x - cut_w // 2, 0, W)
    boundary_x2 = np.clip(center_x + cut_w // 2, 0, W)
    boundary_y1 = np.clip(center_y - cut_h // 2, 0, H)
    boundary_y2 = np.clip(center_y + cut_h // 2, 0, H)

    adjusted_lmbda = 1 - (
        (boundary_x2 - boundary_x1) * (boundary_y2 - boundary_y1)
    ) / (images.size(-1) * images.size(-2))

    random_idx = np.random.permutation(images.size(0))
    shuffled_labels = labels[random_idx]
    new_patches = images[
        random_idx, :, boundary_y1:boundary_y2, boundary_x1:boundary_x2
    ]
    images[:, :, boundary_y1:boundary_y2, boundary_x1:boundary_x2] = new_patches
    return images, shuffled_labels, adjusted_lmbda