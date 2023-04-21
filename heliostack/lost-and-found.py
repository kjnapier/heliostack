def extract(s, threshold=2.0, min_area=3):

    ys, xs = torch.where(s > threshold)
    if len(xs) == 0:
        return [], []
    points = np.array([xs.cpu().numpy(), ys.cpu().numpy()]).T
    clustering = DBSCAN(eps=1, min_samples=min_area, algorithm='ball_tree').fit(points)
    keep = clustering.labels_ > 0
    unique_labels = set(clustering.labels_[keep])
    xs = xs[keep]
    ys = ys[keep]
    points = points[keep]

    x_c = []
    y_c = []
    for k in unique_labels:
        class_member_mask = (clustering.labels_[keep] == k)
        x, y = np.mean(xs[class_member_mask].tolist()), np.mean(ys[class_member_mask].tolist())
        x_c.append(x)
        y_c.append(y)

    return x_c, y_c
