import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs
import itertools


def load_files(authentic_file, impostor_file, ignore_aut=-1, ignore_imp=-1):
    print(f'Loading authentic from {authentic_file}')
    if authentic_file[-4:] == '.txt':
        authentic = np.loadtxt(authentic_file, dtype=np.str)
        print(f'Converting authentic to npy')
        np.save(authentic_file[:-4] + '.npy', authentic.astype(float))
    else:
        authentic = np.load(authentic_file)

    if ignore_aut != -1:
        authentic_score = authentic[authentic[:, 0].astype(int) < ignore_aut, 1].astype(float)

    elif np.ndim(authentic) == 1:
        authentic_score = authentic.astype(float)
    elif np.ndim(authentic) == 2:
        authentic_score = authentic[:, 2].astype(float)
    else:
        authentic_score = authentic[:, 2].astype(float)

    print(f'Loading impostor from {impostor_file}')
    if impostor_file[-4:] == '.txt':
        impostor = np.loadtxt(impostor_file, dtype=np.str)
        print(f'Converting impostor to npy')
        np.save(impostor_file[:-4] + '.npy', impostor.astype(float))
    else:
        impostor = np.load(impostor_file)

    if ignore_imp != -1:
        impostor_score = impostor[impostor[:, 0].astype(int) < ignore_imp, 1].astype(float)

    elif np.ndim(impostor) == 1:
        impostor_score = impostor.astype(float)
    elif np.ndim(impostor) == 2:
        impostor_score = impostor[:, 2].astype(float)
    else:
        impostor_score = impostor[:, 2].astype(float)

    return authentic_score, impostor_score

def compute_dprime(authentic_file1, impostor_file1, l1,
                   authentic_file2, impostor_file2, l2,
                   authentic_file3, impostor_file3, l3,
                   authentic_file4, impostor_file4, l4):

    if l1 is not None and l2 is not None:

        authentic_score1, impostor_score1 = load_files(
            authentic_file1, impostor_file1)
    
        authentic_score2, impostor_score2 = load_files(
            authentic_file2, impostor_file2)


        d_prime1 = (abs(np.mean(authentic_score1) - np.mean(impostor_score1)) /
                np.sqrt(0.5 * (np.var(authentic_score1) + np.var(impostor_score1))))

        d_prime2 = (abs(np.mean(authentic_score2) - np.mean(impostor_score2)) /
                np.sqrt(0.5 * (np.var(authentic_score2) + np.var(impostor_score2))))

        dprime_before_impostor = (abs(np.mean(impostor_score1) - np.mean(impostor_score2)) /
                    np.sqrt(0.5 * (np.var(impostor_score1) + np.var(impostor_score2))))
        
        dprime_before_authentic = (abs(np.mean(authentic_score1) - np.mean(authentic_score2)) /
                    np.sqrt(0.5 * (np.var(authentic_score1) + np.var(authentic_score2))))


        print ("d-prime for first set is: {}".format(d_prime1))
        print ("d-prime for second set is: {}".format(d_prime2))
        print('Before: Delta Impostor d-prime is: {} '.format(dprime_before_impostor))
        print('Before: Delta Authentic d-prime is: {} '.format(dprime_before_authentic))

    if l3 is not None and l4 is not None:
        
        authentic_score3, impostor_score3 = load_files(
            authentic_file3, impostor_file3)
    
        authentic_score4, impostor_score4 = load_files(
            authentic_file4, impostor_file4)


        d_prime3 = (abs(np.mean(authentic_score3) - np.mean(impostor_score3)) /
                np.sqrt(0.5 * (np.var(authentic_score3) + np.var(impostor_score3))))

        d_prime4 = (abs(np.mean(authentic_score4) - np.mean(impostor_score4)) /
                np.sqrt(0.5 * (np.var(authentic_score4) + np.var(impostor_score4))))

        dprime_after_impostor = (abs(np.mean(impostor_score3) - np.mean(impostor_score4)) /
                np.sqrt(0.5 * (np.var(impostor_score3) + np.var(impostor_score4))))
    
        dprime_after_authentic = (abs(np.mean(authentic_score3) - np.mean(authentic_score4)) /
                np.sqrt(0.5 * (np.var(authentic_score3) + np.var(authentic_score4))))

        print ("d-prime for third set is: {}".format(d_prime3))
        print ("d-prime for fourth set is: {}".format(d_prime4))
        print('After: Delta Impostor d-prime is: {} '.format(dprime_after_impostor))
        print('After: Delta Authentic d-prime is: {} '.format(dprime_after_authentic))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Score Histogram')
    parser.add_argument('-authentic1', '-a1', help='Authentic 1 scores.')
    parser.add_argument('-impostor1', '-i1', help='Impostor 1 scores.')
    parser.add_argument('-label1', '-l1', help='Label 1.')
    parser.add_argument('-authentic2', '-a2', help='Authentic 2 scores.')
    parser.add_argument('-impostor2', '-i2', help='Impostor 2 scores.')
    parser.add_argument('-label2', '-l2', help='Label 2.')
    parser.add_argument('-authentic3', '-a3', help='Authentic 3 scores.')
    parser.add_argument('-impostor3', '-i3', help='Impostor 3 scores.')
    parser.add_argument('-label3', '-l3', help='Label 3.')
    parser.add_argument('-authentic4', '-a4', help='Authentic 4 scores.')
    parser.add_argument('-impostor4', '-i4', help='Impostor 4 scores.')
    parser.add_argument('-label4', '-l4', help='Label 4.')

    args = parser.parse_args()

    compute_dprime(args.authentic1, args.impostor1, args.label1,
                   args.authentic2, args.impostor2, args.label2,
                   args.authentic3, args.impostor3, args.label3,
                   args.authentic4, args.impostor4, args.label4)
