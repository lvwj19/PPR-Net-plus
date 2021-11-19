import numpy as np

# ---------------------utils functions------------------------------
def load_obj(file_name):
    with open(file_name, 'r') as f:
        vertexes = []
        faces = []
        for line in f.readlines():
            line = line.strip()
            if line.startswith('v'):
                words = line.split()[1:]
                xyz = [float(w) for w in words]
                vertexes.append(xyz)
            if line.startswith('f'):
                words = line.split()[1:]
                indices = [float(w) for w in words]
                faces.append(indices)
        vertexes = np.array(vertexes)
        faces = np.array(faces, dtype=np.int32)
    return vertexes, faces

def triangle_surface_area(a, b, c):
    s = np.linalg.norm(np.cross((b-a), (c-a)), ord=2)/2.0
    return s

def triangle_surface_center(a, b, c):
    return (a+b+c)/3.0
    

# ---------------------utils functions------------------------------

OBJ_PATH = './SileaneBunny.obj'
# the model unit is m 

if __name__ == "__main__":
    vertexes, faces = load_obj(OBJ_PATH)
    print(vertexes.shape, faces.shape)
    # calculate center of mass 
    center = np.zeros([3])
    total_surface_area = 0
    for face in faces:
        idx_a, idx_b, idx_c = face[0]-1, face[1]-1, face[2]-1
        a, b, c = vertexes[idx_a], vertexes[idx_b], vertexes[idx_c]
        s = triangle_surface_area(a,b,c)
        total_surface_area += s
        center += s * triangle_surface_center(a,b,c)
        # print(triangle_surface_center(a,b,c), s)
        # exit(0)
    center /= total_surface_area
    print('Center of mass: ', center)
    print('S', total_surface_area)

    # calcuate lambda
    vertexes = vertexes - center
    L_sq = np.zeros([3,3])
    for face in faces:
        idx_a, idx_b, idx_c = face[0]-1, face[1]-1, face[2]-1
        a, b, c = vertexes[idx_a], vertexes[idx_b], vertexes[idx_c]
        c_tmp, s_tmp = triangle_surface_center(a,b,c), triangle_surface_area(a,b,c)
        a = a.reshape([3,1])
        b = b.reshape([3 ,1])
        c = c.reshape([3,1])
        c_tmp = c_tmp.reshape([3,1])
        # aaa = np.dot(c_tmp, c_tmp.T)
        theta = s_tmp/12.0*(9.0*np.dot(c_tmp, c_tmp.T)
                +np.dot(a,a.T)+np.dot(b,b.T)+np.dot(c,c.T))
        L_sq += theta
    L = np.sqrt(L_sq/total_surface_area)
    print('lambda: ', L)
    print('lambda_rz: ', np.sqrt(L[0,0]*L[0,0]+L[2,2]*L[2,2]))
