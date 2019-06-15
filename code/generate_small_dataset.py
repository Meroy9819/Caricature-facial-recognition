import csv
import random
import os
import math

IMG_NUM = set()
IMG_FILES = dict()
dir_path = ''
bound = 85 # training bound


def read_csv():
    csv_path = dir_path + '/annotations.csv'
    csv_file = csv.reader(open(csv_path, 'r'))
    print("start read csv...")
    for pic in csv_file:
        img_name = pic[0]
        if img_name == 'image_name':
            continue
        img_type = pic[1]
        img_id = pic[2]
        if int(img_id) < bound:
            img_set = img_id + '_' + img_type
            IMG_NUM.add(img_id)
            if img_set not in IMG_FILES.keys():
                IMG_FILES[img_set] = list()
            else:
                pic_path = '/images/' + img_id + '/' + img_type + '/' + img_name
                IMG_FILES[img_set].append(pic_path)
    print('read csv ok!')


def generate_train_data():
    train_file = dir_path + '/train.txt'
    num = 0
    with open(train_file, 'w') as f1:
        # add matched pictures
        for i in IMG_NUM:
            cartoon_list = str(i) + '_0'
            people_list = str(i) + '_1'
            for cartoon in IMG_FILES[cartoon_list]:
                for people in IMG_FILES[people_list]:
                    train_item = ' '.join([str(i), cartoon, str(i), people, '1']) + '\n'
                    f1.write(train_item)
                    num += 1
        # add unmatched pictures
        i = 0
        while i < num:
            cartoon_list = ''
            people_list = ''
            try:
                first, second = random.sample(IMG_NUM, 2)
                cartoon_list = str(first) + '_0'
                people_list = str(second) + '_1'
                cartoon = random.sample(IMG_FILES[cartoon_list], 1)
                people = random.sample(IMG_FILES[people_list], 1)
                train_item = ' '.join([str(first), cartoon[0], str(second), people[0], '0']) + '\n'
                f1.write(train_item)
                i = i + 1
            except Exception:
                print(cartoon_list)
                print(people_list)

    print('generate training_set ok!')


def generate_test_data():
    temp_file = dir_path + '/test_temp.txt'
    result_file = dir_path+'/test.txt'
    f = open(result_file, "w")
    for line in open(temp_file):
        a = line.split()
        a[0] = "0"
        a[2] = "0"
        temp = ' '.join([a[0], a[1], a[2], a[3], a[4]])+'\n'
        f.write(temp)
    print("generate formal test successful")
    f33 = open(result_file, "rb+")
    f33.seek(-1, os.SEEK_END)
    if f33.next() == "\n":
        f33.seek(-1, os.SEEK_END)
        f33.truncate()
    f33.close()
    if os.path.exists(dir_path + "/test_temp.txt"):
        os.remove(dir_path + "/test_temp.txt")
        print("delete test_temp successful")
    else:
        print("no test_temp exists")


def generate_train_validation_test():
    temp_file = dir_path + '/train.txt'
    result_temp = []
    with open(temp_file, 'r') as f:
        for line in f:
            result_temp.append(list(line.strip('\n').split(',')))
    sum_number_temp = len(result_temp)

    random.seed(10)
    training_site = 0.7
    validation_site = 0.2
    test_site = 0.1
    random.shuffle(result_temp)

    # the percentile of used data
    percentile_data = 1
    result = random.sample(result_temp, int(math.ceil(int(sum_number_temp)*percentile_data)))

    sum_number = len(result)
    train_set_number = int(math.ceil(int(sum_number)*training_site))
    validation_set_number = int(math.ceil(int(sum_number)*validation_site))
    test_set_number = int(math.ceil(int(sum_number)*test_site))
    train_part = result[0:train_set_number]
    validation_part = result[train_set_number:train_set_number+validation_set_number]
    test_part = result[train_set_number+validation_set_number:train_set_number+validation_set_number+test_set_number]

    training_file = dir_path + '/training.txt'
    f1 = open(training_file, 'w')
    for train_item in train_part:
        train_item = ' '.join(train_item) + '\n'
        f1.write(train_item)
    f1.close()
    print('generate train successful!')
    f11 = open(training_file, "rb+")
    f11.seek(-1, os.SEEK_END)
    if f11.next() == "\n":
        f11.seek(-1, os.SEEK_END)
        f11.truncate()
    f11.close()
    validation_file = dir_path + '/validation.txt'
    f2 = open(validation_file, 'w')
    for ver_item in validation_part:
        ver_item = ' '.join(ver_item) + '\n'
        f2.write(ver_item)
    f2.close()
    print('generate validation successful')
    f22 = open(validation_file, "rb+")
    f22.seek(-1, os.SEEK_END)
    if f22.next() == "\n":
        f22.seek(-1, os.SEEK_END)
        f22.truncate()
    f22.close()

    test_file = dir_path + '/test_temp.txt'
    f3 = open(test_file, 'w')
    for test_item in test_part:
        test_item = ' '.join(test_item) + '\n'
        f3.write(test_item)
    f3.close()
    print('generate test temp successful')


def distribution(file_path, m_type, file_number):
    f = open(file_path, "r")
    lines = f.readlines()
    length = len(lines)
    part_length = math.ceil(length/file_number)
    content = ""
    j = 1
    for i in range(1, length):
        if i % part_length != 0:
            content += lines[i-1]
        else:
            content += lines[i-1]
            new_file_name = '/' + m_type + '_' + str(j) + '.txt'
            j = j + 1
            f2 = open(dir_path + new_file_name, 'w')
            f2.write(content.strip('\n'))
            content = ""
            f2.close()
    f.close()


def generate_data(path, nb_class=85):
    global dir_path
    dir_path = path
    global bound
    bound = nb_class
    read_csv()
    generate_train_data()
    generate_train_validation_test()
    generate_test_data()

if __name__ == '__main__':
    generate_data('./af2019-ksyun-training-20190416', 85)
