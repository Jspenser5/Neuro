def read_file_datas(text_data):
    file = open(text_data)
    lines = file.readlines()
    filepaths = []
    labels = []
    for line in lines:
        splitted = line.rstrip().split()
        filepaths.append(splitted[0])
        labels.append(splitted[1])
    return (filepaths, labels)