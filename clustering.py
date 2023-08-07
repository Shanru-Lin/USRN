import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import datetime
import argparse
import json

def main(conf, clustering_algorithm):

    model_folder = config['trainer']['save_dir'] + config['experim_name']

    # example: a specific data point that you are clustering. each has associated features (attributes) that are used to characterize it.
    # features: a variable representing each example in a multi-dimensional space, loaded from files.
            # Features extracted from different examples/images" refers to the process of computing numerical representations 
            # that capture specific characteristics or patterns present in images.
    # target: created by resizing the corresponding label data using OpenCV, and it is used in the clustering process to group similar examples together.

    # a step-by-step breakdown of how these terms are used in your code:
    # 1. For each file (representing an example) in the dataset:
    #   Load the feature data (feat) from the file.
    #   Process the label data to create the target assignments (target).
    #   Append the feature data and target assignments to the respective arrays (feats and targets).
    #   Assign an identifier to the example using file_id.
    # 2. The feats array holds all the feature data for all examples, and the targets array holds the corresponding target assignments.
    # 3. Depending on the clustering algorithm (clustering_algorithm), you perform clustering on the targets array to group similar examples together into clusters. The specific algorithm determines how these clusters are formed.
    # 4. During clustering, the target assignments for some examples may change to reflect their new cluster assignments.
    # The combination of features and target assignments allows you to apply the clustering algorithm and group similar examples together based on their feature similarity. The clustering result is stored in the targets_subcls array, which holds the final cluster assignments for each example.


    # split_list is a list that contains the number of clusters you want to create for each parent class (or cluster) in your clustering algorithm
    ### VOC Dataset
    if config['dataset'] == 'voc':
        label_folder = 'datasets/voc/VOCdevkit/VOC2012/SegmentationClassAug'
        if conf['n_labeled_examples'] == 662:
            split_list = [132, 2, 1, 1, 1, 2, 3, 4, 7, 2, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 331:
            split_list = [121, 2, 1, 1, 1, 1, 3, 3, 6, 3, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 165:
            split_list = [136, 2, 2, 1, 1, 1, 2, 4, 8, 3, 1, 2, 7, 2, 2, 18, 1, 1, 1, 3, 3]
    ### Cityscapes Dataset
    elif config['dataset'] == 'cityscapes':
        label_folder = 'datasets/cityscapes/segmentation/train'
        if conf['n_labeled_examples'] == 372:
            split_list = [42, 7, 26, 1, 2, 2, 1, 1, 19, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 186:
            split_list = [45, 7, 28, 1, 2, 2, 1, 1, 20, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 93:
            split_list = [38, 6, 22, 1, 2, 2, 1, 1, 17, 2, 5, 1, 1, 7, 1, 1, 1, 1, 1]

    save_folder = os.path.join(model_folder, 'label_subcls_' + clustering_algorithm)
    os.makedirs(save_folder, exist_ok=True)
    feature_folder = os.path.join(model_folder, 'features')
    # #{
    # os.makedirs(feature_folder, exist_ok=True)
    # #}

    subclasses =  np.cumsum(np.asarray(split_list)) # cumulative sum of elements in the array
    subclasses = np.insert(subclasses, 0, 0) # prepend 0 to subclasses array  to account for the fact that there are no subclasses for the initial parent class (class index 0)
    
    # Here's an example to illustrate the purpose of subclasses:
    # Suppose split_list is [3, 2, 4, 2], which means you want to create 3, 2, 4, and 2 clusters for the first four parent classes. After the calculations above, 
    # the subclasses array might look like [0, 3, 5, 9, 11]. This means:
    # Parent class 0 has 3 subclasses (starting index 0).
    # Parent class 1 has 2 subclasses (starting index 3).
    # Parent class 2 has 4 subclasses (starting index 5).
    # Parent class 3 has 2 subclasses (starting index 9).
    
    # This information is useful when assigning the correct subclass indices to examples during the clustering process, 
    # as each parent class's subclasses should be numbered consecutively.
    # Later in your code, you use subclasses[cls] to determine the starting index for subclasses of a particular parent class when generating cluster assignments. 
    # This ensures that the assignments are consistent with the overall clustering plan and organization.
    
    oldtime=datetime.datetime.now()
    files = os.listdir(feature_folder)
    list.sort(files)
    feat_shape_list = []
    label_shape_list = []
    for i, file in enumerate(files):
        feat = np.load(os.path.join(feature_folder, file))
        feat_shape_list.append(feat.shape[-2:])
        H, W = feat.shape[-2], feat.shape[-1]
        # Load label data based on the dataset type 
        if config['dataset'] == 'cityscapes':
            label = np.asarray(Image.open(os.path.join(label_folder, file.replace('_leftImg8bit.npy', '_gtFine_labelTrainIds.png'))))
        else:
            label = np.asarray(Image.open(os.path.join(label_folder, file.replace('.npy', '.png'))))
        label_shape_list.append(label.shape[-2:])
        # Resize the label to match the dimensions of the feature array using nearest-neighbor interpolation. 
        # This operation results in a new 'target' array that corresponds to the resized label.
        target = cv2.resize(label, (W,H), interpolation=cv2.INTER_NEAREST)
        feat = feat.reshape(feat.shape[0],-1).transpose((1,0))
        target = np.expand_dims(target.reshape(-1), axis=1)
        if i==0:
            feats = feat
            targets = target
            file_id = i * np.ones(target.shape)
        else:
            feats = np.vstack((feats, feat))
            targets = np.vstack((targets, target))
            file_id = np.vstack((file_id, i * np.ones(target.shape)))

    newtime=datetime.datetime.now()
    print('data_loading: %s'%(newtime-oldtime))
    
    print(Counter(targets.reshape(-1).tolist()))

    if clustering_algorithm == 'normal_kmeans':
        targets_subcls = targets.copy()
        for cls in np.unique(targets):
            print('Parent class:', cls)
            oldtime = datetime.datetime.now()
            if cls < 255: # Checking Validity
                num_clusters = split_list[cls] # Determining Number of Clusters (of the current parent class)
                subcls = subclasses[cls] # Assigning Starting Index of Subclass (of the current parent class) among all subclasses
                if num_clusters == 1:
                    targets_subcls[targets==cls] = subcls
                    # set all targets belonging to the current parent class to the corresponding subclass index subcls. 
                    # This means all pixels in this class will be assigned to a single cluster.
                else:
                    subindex = np.where(targets==cls)[0]
                    #  contains the indices of examples or images that belong to the current parent class cls.
                    subfeats = feats[subindex,:]
                    # this line extracts the corresponding rows from the feats array. 
                    # These rows contain the features of the examples or images that belong to the current parent class cls.
                    k_center = MiniBatchKMeans(n_clusters=num_clusters, random_state=0).fit(subfeats)
                    newtime = datetime.datetime.now()
                    print('KMeans: %s' % (newtime - oldtime))
                    lbls = k_center.labels_
                    # Each element in the lbls array corresponds to a data point in subfeats and indicates which subcluster that data point belongs to.
                    for j in range(num_clusters):
                        targets_subcls[subindex[lbls==j]] = subcls + j
                    '''
                    # An Example:
                    # targets_subcls = [10, 10, 10, 10, 10, 10, 10]

                    # # Loop over subclusters (j = 0, 1, 2)
                    # for j in range(3):
                    #     # Get indices of examples/images belonging to subcluster j within class 2
                    #     indices = subindex[lbls == j]
                    #     # Update targets_subcls array for these indices
                    #     targets_subcls[indices] = subcls + j

                    # # Updated targets_subcls array after clustering
                    # targets_subcls = [11, 10, 12, 11, 12, 10, 11]
                    '''
    elif clustering_algorithm == 'balanced_kmeans':
        targets_subcls = targets.copy()
        for cls in np.unique(targets):
            print('Parent class:', cls)
            if cls < 255:
                num_clusters = split_list[cls] # total number of subclasses (of the current parent class)
                print("num_clusters: ", num_clusters)
                subcls = subclasses[cls] # Starting Index of Subclass (of the current parent class)
                if num_clusters == 1:
                    targets_subcls[targets == cls] = subcls
                else:
                    subindex = np.where(targets == cls)[0] # indices of images of the current parent class
                    subfeats = feats[subindex, :] # extracts the corresponding rows from the feats array

                    data_int16_x1000 = np.int16(subfeats * 1000)
                    np.savetxt(save_folder + '/subfeats_cls' + str(cls) + '_n' + str(num_clusters) + '.csv', data_int16_x1000,
                               fmt='%i', delimiter=',')
                    print("feats saved")

                    #{original command}
                    command = "regularized-k-means/build/regularized-k-means hard "+ save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + ".csv " + str(num_clusters) + \
                              " -a " + save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + "_hard_assignments -o "+ save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + "_hard_summary.txt -t-1"
                    print(command)
                    os.system(command)

                    lbls = np.loadtxt(save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + "_hard_assignments.csv", delimiter=',')

                    print(Counter(lbls.reshape(-1).tolist()))

                    for j in range(num_clusters):
                        targets_subcls[subindex[lbls == j]] = subcls + j

    for i, file in enumerate(files):
        # each file representing features from one image
        tgt_subcls = targets_subcls[file_id==i]
        feat_shape = feat_shape_list[i]
        tgt_subcls = tgt_subcls.reshape(feat_shape)
        H, W = label_shape_list[i]
        tgt_subcls = cv2.resize(tgt_subcls, (W,H), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(tgt_subcls).save(os.path.join(save_folder, file.replace('.npy','.png')))
    print(Counter(targets_subcls.reshape(-1).tolist()))

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/voc_1over32_baseline.json', type=str,)
    parser.add_argument('-ca', '--clustering_algorithm', default='balanced_kmeans', type=str,
                        help="Support 'balanced_kmeans' or 'normal_kmeans'")
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config, args.clustering_algorithm)