import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('result')
parser.add_argument('--ylabel', default='')
args = parser.parse_args()


def show(step_list, class_f1_map, y_label):
    save_path = './{}_result.png'.format(y_label)
    plt.figure(figsize=(10, 5))
    plt.title('{} result analyse'.format(y_label))
    plt.xlabel('step')
    plt.ylabel(y_label)
    lines = []
    label_names = []
    max_step_list = []
    max_value_list = []
    for label, value_list in class_f1_map.items():
        # print max value in list
        max_index = value_list.index(max(value_list))
        print('{} get max value {} at {} step.'.format(label, value_list[max_index], step_list[max_index]))
        max_step_list.append(step_list[max_index])
        max_value_list.append(value_list[max_index])

        x_list = [int(n) for n in step_list]
        x_y = [[i,j] for i,j in zip(x_list, value_list)]
        sorted_x_y = sorted(x_y, key=lambda x: x[0])
        new_x, new_y = [], []
        for x_y in sorted_x_y:
            new_x.append(x_y[0])
            new_y.append(x_y[1])

        line = plt.plot(new_x, new_y)
        lines.append(line)
        label_names.append(label)
    plt.legend(lines, labels=label_names, loc='best')
    plt.plot(max_step_list, max_value_list, 'ro')
    plt.savefig(save_path)
    plt.show()


def main():
    result_path = args.result
    if args.ylabel:
        y_label = args.ylabel
    else:
        y_label = os.path.basename(result_path).split('.')[0].split('_')[0]
    data = pd.read_csv(result_path)
    print(data.columns)
    kv_map = {}
    for i in data.columns:
        if i == 'Unnamed: 0':
            continue
        kv_map[i] = list(data[i])
    step_list = kv_map.pop('step')
    show(step_list, kv_map, y_label)
        


if __name__ == '__main__':
    main()

