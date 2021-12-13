
def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for index, name in enumerate(data):
            names[index] = name.strip('\n')
    return names
