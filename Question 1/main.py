# problem format: [m, n, sum tesrget sequence... ]
pset = [[9, 9, 65, 72, 90, 110],
        [90000, 100000, 5994891682]]

with open('output_question_1.txt','w') as f1:
    for problem in pset:
        m = problem[0]
        n = problem[1]
        sum_list = problem[2:]
        for target in sum_list:
            if target<((1+m)*m/2+n-1) or target>((1+m)*m/2+(n-1)*m): 
                raise Exception('invalid sum target')
            Rtarget = target - (1+m)*m/2
            c1num = int(Rtarget // (n-1))
            c2num = c1num + 1
            c2steps = int(Rtarget - c1num*(n-1))
            c1steps = int(n - 1 - c2steps)
            oper = 'D'*(c1num-1) + 'R'*int(c1steps) + \
                    'D'*(c2num-c1num) + 'R'*(c2steps) + \
                    'D'*(m+n-2-(c1num-1)-c1steps-(c2num-c1num)-c2steps)
            f1.write(str(target) + ' ' + oper + '\n')
        f1.write('\n')

        
