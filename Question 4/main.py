import numpy as np 

def read_input(file_name):
    with open(file_name, 'r') as f1:
        content = f1.read()
        content = content.split('\n')[:-1]
        m = len(content)
        content = ('\t'.join(content)).split('\t')
        out = np.array([int(a) for a in content])
        return out.reshape([m, -1])

def write_output(file_name, mat):
    with open(file_name, 'w') as f1:
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                f1.write(str(mat[row,col]))
                if col!=mat.shape[1]-1:
                    f1.write('\t')
            f1.write('\n')

        
def find_cnct(loc, mat, label):
    i = loc[0]
    j = loc[1]
    if mat[i,j]==-1:
        mat[i,j]=label
        if i>0:
            mat = find_cnct([i-1,j], mat, label)
        if j>0:
            mat = find_cnct([i,j-1], mat, label)
        if i<mat.shape[0]-1:
            mat = find_cnct([i+1,j], mat, label)
        if j<mat.shape[1]-1:
            mat = find_cnct([i,j+1], mat, label)
    return mat


if __name__=='__main__':
    my_input = read_input('input_question_4.txt')
    my_output = -my_input
    m, n = my_input.shape
    label = 1
    for ii in range(m):
        for jj in range(n):
            if my_output[ii,jj]==-1:
                my_output = find_cnct([ii,jj], my_output, label)
                label += 1
    write_output('output_question_4.txt', my_output)
    
    
    #print(my_output)
    #print('Press \'Enter\' to quit.')
    #input()
