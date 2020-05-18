import numpy as np
def read_coordinate(file_name):
    with open(file_name, 'r') as f1:
        content = f1.read()
        content = content.split('\n')[1:-1]
        content = ('\t'.join(content)).split('\t')
        out = np.array([int(a) for a in content])
        return out.reshape([-1, 2])

def read_index(file_name):
    with open(file_name, 'r') as f1:
        content = f1.read()
        content = content.split('\n')[1:-1]
        out = np.array([int(a) for a in content])
        return out.reshape([-1, 1])

def write_coordinate(file_name, coor):
    with open(file_name, 'w') as f1:
        f1.write('x1\tx2\n')
        for row in range(coor.shape[0]):
            f1.write(str(coor[row,0]) + '\t' + str(coor[row,1]) + '\n')

def write_index(file_name, index):
    with open(file_name, 'w') as f1:
        f1.write('index\n')
        for row in range(index.shape[0]):
            f1.write(str(index[row,0]) + '\n')

def coor_to_index(coor, dim):
    index = coor[:,0] + coor[:,1]*dim[0]
    return index.reshape([-1,1])

def index_to_coor(index, dim):
    x1 = index % dim[0]
    x2 = index // dim[0]
    return np.hstack([x1, x2])

if __name__ == '__main__':
    dim = [50, 57]
    coor_input = read_coordinate('input_coordinates_7_1.txt')
    index_output = coor_to_index(coor_input, dim)
    write_index('output_index_7_1.txt', index_output)
    index_input = read_index('input_index_7_1.txt')
    coor_output = index_to_coor(index_input, dim)
    print(coor_output)
    write_coordinate('output_coordinates_7_1.txt', coor_output)
