import numpy as np
from sklearn.cluster import OPTICS,DBSCAN
from rearrange_on_proc.constants import CATEGORY_LIST, CATEGORY_to_ID



def point_cloud_instance_seg(point_cloud,feature_list):
    '''
    point_cloud_divided_by_instance = {
        '{categoryA}':[
            {
                "points": np.ndarray([n,3]), #x,y,z
                "feature":np.ndarrat([m])
            },
            ...,
            {
                "points": np.ndarray([n,3]), #x,y,z
                "feature":np.ndarrat([m])
            }
        ],
        '{categoryB}':[
            {
                "points": np.ndarray([n,3]), #x,y,z
                "feature":np.ndarrat([m])
            },
            ...
            {
                "points": np.ndarray([n,3]), #x,y,z
                "feature":np.ndarrat([m])
            }
        ],
        ...
    }
    '''
    point_cloud_divided_by_instance = {}
    for category in CATEGORY_to_ID.keys():
        category_id = CATEGORY_to_ID[category]
        category_point_cloud = point_cloud[point_cloud[:,4]==category_id]
        category_point_cloud_xyz = category_point_cloud[:,:3]
        if category_point_cloud.shape[0] <= 10:
            continue
        # clustering = OPTICS(min_samples=10).fit(category_point_cloud_xyz)
        clustering = DBSCAN(eps=0.10, min_samples=5).fit(category_point_cloud_xyz)
        unique_clusters = np.unique(cluster_labels)
        # point_cloud_visualization(category_point_cloud_xyz,cluster_labels,category)
        for label in unique_clusters:
            if label == -1:
                continue
            instance_point_cloud = category_point_cloud[cluster_labels == label]
            indices = instance_point_cloud[:,5].astype(int)
            selected_feature = np.array([feature_list[i] for i in indices])
            fuse_feature = selected_feature.mean(axis=0)
            if category not in point_cloud_divided_by_instance:
                point_cloud_divided_by_instance[category] = {}
            new_instance_id = category + '_' + str(len(point_cloud_divided_by_instance[category]) + 1)
            point_cloud_divided_by_instance[category][new_instance_id] = {"points":instance_point_cloud[:,:3],"feature":fuse_feature}
    return point_cloud_divided_by_instance

def point_cloud_seg_visualization(points,labels,category):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='viridis', s=50)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster Label')

    ax.set_title('DBSCAN Clustering in 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f"./test/point_cloud/{category}.jpg")
    plt.close()

