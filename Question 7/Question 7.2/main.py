import numpy as np

def read_coordinate(file_name, dim):
    with open(file_name, 'r') as f1:
        content = f1.read()
        content = content.split('\n')[1:-1]
        content = ('\t'.join(content)).split('\t')
        out = np.array([int(a) for a in content])
        return out.reshape([-1, len(dim)])

def read_index(file_name):
    with open(file_name, 'r') as f1:
        content = f1.read()
        content = content.split('\n')[1:-1]
        out = np.array([int(a) for a in content])
        return out.reshape([-1, 1])

def write_coordinate(file_name, coor):
    with open(file_name, 'w') as f1:
        for col in range(coor.shape[1]):
            f1.write('x' + str(col+1))
            if col!=coor.shape[1]-1:
                f1.write('\t')
        f1.write('\n')
        for row in range(coor.shape[0]):
            for col in range(coor.shape[1]):
                f1.write(str(coor[row,col]))
                if col!=coor.shape[1]-1:
                    f1.write('\t')
            f1.write('\n')
            

def write_index(file_name, index):
    with open(file_name, 'w') as f1:
        f1.write('index\n')
        for row in range(index.shape[0]):
            f1.write(str(index[row,0]) + '\n')

def coor_to_index(coor, dim):
    index = 0
    for ii in range(len(dim)):
        temp = coor[:,ii]
        for jj in range(ii):
            temp *= dim[jj]
        index += temp
    return index.reshape([-1,1])

def index_to_coor(index, dim):
    coor = np.ndarray([index.shape[0],0])
    for nn in range(len(dim)):
        temp = index[:]
        # mod
        L_acc = 1
        for ii in range(nn+1):
            L_acc *= dim[ii]
        temp = temp % L_acc
        # divide
        L_acc = 1
        for jj in range(nn):
            L_acc *= dim[jj]
        temp = temp // L_acc

        coor = np.hstack([coor, temp])
    return coor

if __name__ == '__main__':
    dim = [4, 8, 5, 9, 6, 7]
    coor_input = read_coordinate('input_coordinates_7_2.txt', dim)
    index_output = coor_to_index(coor_input, dim)
    write_index('output_index_7_2.txt', index_output)
    index_input = read_index('input_index_7_2.txt')
    coor_output = index_to_coor(index_input, dim)
    write_coordinate('output_coordinates_7_2.txt', coor_output.astype(int))
