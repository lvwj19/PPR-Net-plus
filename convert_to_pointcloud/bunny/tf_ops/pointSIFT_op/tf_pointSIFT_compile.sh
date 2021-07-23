#/bin/bash
/usr/local/cuda-9.0/bin/nvcc pointSIFT.cu -o pointSIFT_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# TF1.2
#g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I /home/likewise-open/SENSETIME/dongzhikai/tf_env/tf_env/local/lib/python2.7/site-packages/tensorflow/include -I/usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I /home/idesignlab/anaconda3/envs/pointnet2/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /home/idesignlab/anaconda3/envs/pointnet2/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/home/idesignlab/anaconda3/envs/pointnet2/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


