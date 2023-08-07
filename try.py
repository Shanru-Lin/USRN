# save_folder = "saved/voc_1over32_baseline/label_subcls_balanced_kmeans/subfeats_cls"
# cls = 0
# num_clusters = 121
# command = "cd regularized-k-means"
# command = "build/regularized-k-means hard "+ "../"+ save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + ".csv " + str(num_clusters) + \
#                             " -a " + save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + "_hard_assignments -o"+ save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) + "_hard_summary.txt -t 20"
# print(command)
# #regularized-k-means/build/regularized-k-means hard saved/voc_1over32_baseline/label_subcls_balanced_kmeans/subfeats_cls0_n121.csv 121 -a saved/voc_1over32_baseline/label_subcls_balanced_kmeans/subfeats_cls0_n121_hard_assignments -o saved/voc_1over32_baseline/label_subcls_balanced_kmeans/subfeats_cls0_n121_hard_summary.txt -t 20
import os
# command = "cd try; touch a.txt"
# os.system(command)
# # command = "touch a.txt"
# # os.system(command)
save_folder = "saved/voc_1over32_baseline/label_subcls_balanced_kmeans"
cls = 0
num_clusters = 121

dir_first_half = save_folder + "/subfeats_cls" + str(cls) + "_n" + str(num_clusters) 
in_dir = dir_first_half + ".csv"
arg_dir = dir_first_half + "_hard_assignments"
out_dir = dir_first_half + "_hard_summary.txt"
# assume the current directory is USRN
os.chdir("regularized-k-means")
command = "build/regularized-k-means hard "+in_dir+" "+str(num_clusters)+" -a "+arg_dir+" -o "+out_dir+" -t20"
print(command)

# os.chdir("../try2")
# command = "touch d.txt"
# os.system(command)