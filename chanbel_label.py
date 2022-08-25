import numpy as np
import os

 
def main():
    file_path="/home/bi/study/thesis/data/synthetic/finetune"
    List_motor = os.listdir(file_path)
    if 'display.py' in List_motor :
        List_motor.remove('display.py')
    if '.DS_Store' in List_motor :
        List_motor.remove('.DS_Store')
    List_motor.sort()
    cnt=1
    for dirs in List_motor :
        Motor_path = file_path + '/' + dirs
        # if "98" in dirs:
        if True:
            if dirs.split('.')[1]=='txt':
            
                patch_motor=np.loadtxt(Motor_path)   
            else:
                patch_motor=np.load(Motor_path)  
        size=patch_motor.shape[0]
        # cnt=0
        for i in range(size):
            if patch_motor[i][3]==6:
                patch_motor[i][3]=5
        # cnt+=1
        print(cnt)
        print(str(cnt) + " processed ")
        np.save(Motor_path,patch_motor)
        cnt=cnt+1

if __name__ == '__main__':
    main()