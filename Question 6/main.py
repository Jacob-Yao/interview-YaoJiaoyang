import numpy as np 

def read_input(file_name):
    with open(file_name, 'r') as f1:
        content = f1.read()
        content = content.split('\n')[:-1]
        content = (' '.join(content)).split(' ')
        out = np.array([int(a) for a in content])
        return out.reshape([-1, 2])
        
def write_output(file_name, points, judgement):
    with open(file_name, 'w') as f1:
        for row in range(points.shape[0]):
            f1.write(str(points[row,0])+' '+str(points[row,1])+'\t')
            if judgement[row]==1:
                f1.write('inside\n')
            else:
                f1.write('outside\n')

def on_segment(P1, P2, Q):
    if (Q[1]-P1[1])*(Q[1]-P2[1])>0:
        return False
    if (Q[0]-P1[0])*(Q[0]-P2[0])>0:
        return False
    if np.cross(Q-P1, Q-P2)==0:
        return True
    else: 
        return False

def pass_segment(P1, P2, Q):
    if Q[1]>=P1[1] and Q[1]>=P2[1]:
        return False
    if Q[1]<P1[1] and Q[1]<P2[1]:
        return False
    if P1[0]==P2[0]:
        if Q[0]<= P1[0]:
            return True
        else: 
            return False
    k = (P1[1] - Q[1]) / (P1[1] - P2[1])
    if  P1[0] + k * (P2[0] - P1[0]) >= Q[0]:
        return True
    else:
        return False

def in_polygon(polygon, Q):
    P_list = np.vstack([polygon,polygon[0]])
    pass_time = 0
    for idx in range(polygon.shape[0]):
        P1 = P_list[idx]
        P2 = P_list[idx+1]
        if on_segment(P1, P2, Q):
            return True
        if pass_segment(P1, P2, Q):
            pass_time += 1
    print(pass_time)
    if pass_time%2==0:
        return False
    else: 
        return True

if __name__ == '__main__':
    polygon = read_input('input_question_6_polygon.txt')
    Q_list = read_input('input_question_6_points.txt')
    judge_list = [False] * Q_list.shape[0]
    for idx in range(Q_list.shape[0]):
        judge_list[idx] = in_polygon(polygon, Q_list[idx])
    write_output('output_question_6.txt', Q_list, judge_list)