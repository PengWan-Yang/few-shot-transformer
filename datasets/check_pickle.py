import pickle,os
_pickle_file = 'train_data_3fps_flipped.pkl'
file = open(_pickle_file, 'rb')
# dump information to that file
data = pickle.load(file)

for ele in data:
    if 'C40k' in ele['fg_name']:
        print(ele['fg_name'])
    if not os.path.exists(ele['fg_name'] ):
        print(ele['fg_name'])
