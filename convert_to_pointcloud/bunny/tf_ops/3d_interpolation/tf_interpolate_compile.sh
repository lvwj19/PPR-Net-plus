
# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/idesignlab/anaconda3/envs/pointnet2/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /home/idesignlab/anaconda3/envs/pointnet2/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/home/idesignlab/anaconda3/envs/pointnet2/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
