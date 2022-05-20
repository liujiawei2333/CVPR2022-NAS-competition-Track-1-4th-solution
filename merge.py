import json
import argparse


parser = argparse.ArgumentParser(description='subnet merge')
parser.add_argument('--epoch', default='4_all_test', type=str)
parser.add_argument('--save', default='0516', type=str)
parser.add_argument('--stage',default='3_from_2',type=str)
parser.add_argument('--top', default=1, type=int)
run_args = parser.parse_args()


save = run_args.save
epoch = run_args.epoch
stage = run_args.stage
top = run_args.top

save_file1 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json1.json'
save_file2 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json2.json'
save_file3 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json3.json'
save_file4 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json4.json'
save_file5 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json5.json'
save_file6 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json6.json'
save_file7 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json7.json'
save_file8 = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json8.json'

save_file_final = './results/' + save + '/' + stage + '/json_results/epoch'+str(epoch) + '/top'+str(top)+'_json_final.json'

with open(save_file1) as save_file1:
        save_file1 = json.load(save_file1)

with open(save_file2) as save_file2:
        save_file2 = json.load(save_file2)

with open(save_file3) as save_file3:
        save_file3 = json.load(save_file3)

with open(save_file4) as save_file4:
        save_file4 = json.load(save_file4)

with open(save_file5) as save_file5:
        save_file5 = json.load(save_file5)

with open(save_file6) as save_file6:
        save_file6 = json.load(save_file6)

with open(save_file7) as save_file7:
        save_file7 = json.load(save_file7)

with open(save_file8) as save_file8:
        save_file8 = json.load(save_file8)

json_out_dict = save_file1.copy()
json_out_dict.update(save_file2)
json_out_dict.update(save_file3)
json_out_dict.update(save_file4)
json_out_dict.update(save_file5)
json_out_dict.update(save_file6)
json_out_dict.update(save_file7)
json_out_dict.update(save_file8)


with open(save_file_final, "w") as f:
        json.dump((json_out_dict), f)